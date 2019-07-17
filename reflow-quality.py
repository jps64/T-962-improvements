#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Log the temperatures reported by the oven in a live plot and
# in a CSV file.
#
# Requires
# python 2.7
# - pyserial (python-serial in ubuntu, pip install pyserial)
# - matplotlib (python-matplotlib in ubuntu, pip install matplotlib)
#

import csv
import datetime
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import matplotlib.gridspec as gspec
import serial
import sys
from time import time
import numpy as np

# global settings
#
FIELD_NAMES = 'Time,Temp0,Temp1,Temp2,Temp3,Set,Actual,Heat,Fan,ColdJ,Mode'
TTYs = ('/dev/tty.usbserial-A800HCV2', '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2')
BAUD_RATE = 115200

TEMPERATURE_NAMES = 'TSTART,TSMIN,TSMAX,TLIQPRE,TPEAKPRE,TPEAK,TPEAKPOST,TLIQPOST,TEND'
temperature_leaded = dict(zip(TEMPERATURE_NAMES.split(','), \
		[ 60.0,100.0,150.0,183.0,230.0,235.0,230.0,183.0,100.0]))
temperature_leadfree = dict(zip(TEMPERATURE_NAMES.split(','), \
		[ 60.0,150.0,200.0,217.0,255.0,260.0,255.0,217.0,100.0]))
temperature_profile = temperature_leaded
temperature_direction = dict(zip(TEMPERATURE_NAMES.split(','), \
		[1,1,1,1,1,0,-1,-1,-1]))


QUALITY_NAMES  = 'tPRE,tLIQUID,tPOST,TMAX,dTUP,dTDOWN'
quality_min =	dict(zip(QUALITY_NAMES.split(','),  [ 60.0, 25.0, 17.0,188.0,  0.5,  -6.0]))
quality_max =	dict(zip(QUALITY_NAMES.split(','),  [240.0,150.0,180.0,250.0,  3.0,  -0.5]))
quality_optmin = dict(zip(QUALITY_NAMES.split(','), [100.0, 60.0, 70.0,210.0,  1.7,  -4.6]))
quality_optmax = dict(zip(QUALITY_NAMES.split(','), [180.0, 90.0, 90.0,225.0,  2.3,  -3.4]))
quality_threshold = 0.8

logdir = 'logs/'

MAX_X = 480
MAX_Y_temperature = 300
MAX_Y_pwm = 260
#
# end of settings

#------------------------------------------------------------------------------
def radar_factory(num_vars, frame='circle'):
	"""Create a radar chart with `num_vars` axes.

	This function creates a RadarAxes projection and registers it.

	Parameters
	----------
	num_vars : int
		Number of variables for radar chart.
	frame : {'circle' | 'polygon'}
		Shape of frame surrounding axes.

	"""
	# calculate evenly-spaced axis angles
	theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

	def draw_poly_patch(self):
		# rotate theta such that the first axis is at the top
		verts = unit_poly_verts(theta + np.pi / 2)
		return plt.Polygon(verts, closed=True, edgecolor='k')

	def draw_circle_patch(self):
		# unit circle centered on (0.5, 0.5)
		return plt.Circle((0.5, 0.5), 0.5)

	patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
	if frame not in patch_dict:
		raise ValueError('unknown value for `frame`: %s' % frame)

	class RadarAxes(PolarAxes):

		name = 'radar'
		# use 1 line segment to connect specified points
		RESOLUTION = 1
		# define draw_frame method
		draw_patch = patch_dict[frame]

		def __init__(self, *args, **kwargs):
			super(RadarAxes, self).__init__(*args, **kwargs)
			# rotate plot such that the first axis is at the top
			self.set_theta_zero_location('N')

		def fill(self, *args, **kwargs):
			"""Override fill so that line is closed by default"""
			closed = kwargs.pop('closed', True)
			return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

		def plot(self, *args, **kwargs):
			"""Override plot so that line is closed by default"""
			lines = super(RadarAxes, self).plot(*args, **kwargs)
			for line in lines:
				self._close_line(line)

		def _close_line(self, line):
			x, y = line.get_data()
			# FIXME: markers at x[0], y[0] get doubled-up
			if x[0] != x[-1]:
				x = np.concatenate((x, [x[0]]))
				y = np.concatenate((y, [y[0]]))
				line.set_data(x, y)

		def set_varlabels(self, labels):
			self.set_thetagrids(np.degrees(theta), labels)

		def _gen_axes_patch(self):
			return self.draw_patch()

		def _gen_axes_spines(self):
			if frame == 'circle':
				return PolarAxes._gen_axes_spines(self)
			# The following is a hack to get the spines (i.e. the axes frame)
			# to draw correctly for a polygon frame.

			# spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
			spine_type = 'circle'
			verts = unit_poly_verts(theta + np.pi / 2)
			# close off polygon by repeating first vertex
			verts.append(verts[0])
			path = Path(verts)

			spine = Spine(self, spine_type, path)
			spine.set_transform(self.transAxes)
			return {'polar': spine}

	register_projection(RadarAxes)
	return theta

#------------------------------------------------------------------------------
def unit_poly_verts(theta):
	"""Return vertices of polygon for subplot axes.

	This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
	"""
	x0, y0, r = [0.5] * 3
	verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
	return verts

#------------------------------------------------------------------------------
def gt_pos(a=[], reference=0.0):
	if (len(a) == 0): return -1
	for index, value in enumerate(a):
		if (value >= reference): return index
	return -1

#------------------------------------------------------------------------------
def gt_pos_rev(a=[], reference=0.0):
	if (len(a) == 0): return -1
	for index, value in enumerate(reversed(a)):
		if (value >= reference): return (len(a) - index - 1)
	return -1

#------------------------------------------------------------------------------
def timestamp(dt=None):
	if dt is None:
		dt = datetime.datetime.now()

	return dt.strftime('%Y-%m-%d-%H%M%S')

#------------------------------------------------------------------------------
def logname(filetype, profile):
	return '%s%s-%s.%s' % (
		logdir,
		timestamp(),
		profile.replace(' ', '_').replace('/', '_'),
		filetype
	)

#------------------------------------------------------------------------------
def get_tty():
	for devname in TTYs:
		try:
			port = serial.Serial(devname, baudrate=BAUD_RATE)
			print ('Using serial port %s' % port.name)
			return port

		except:
			print ('Tried serial port %s, but failed.' % str(devname))
			pass

	return None

#------------------------------------------------------------------------------
class Line(object):
	def __init__(self, pane, key, label=None):
		self.xvalues  = []
		self.yvalues  = []
		self.dyvalues = []
		self.key = key
		self.temp_max = 0.0
		self.temp_max_pos = 0
		self.quality = 0.0

		self._line, = pane.plot(self.xvalues, self.yvalues, label=label or key)

	def add(self, log):
		self.xvalues.append(log['Time'])
		self.yvalues.append(log[self.key])

		self.update()

	def update(self):
		self._line.set_data(self.xvalues, self.yvalues)

	def clear(self):
		self.xvalues = []
		self.yvalues = []

		self.update()

	def derivation(self, dist=5):		# argument dist shall be odd
		length = len(self.yvalues)
		imax = length - 1
		step = int(dist / 2)
		if (length >= dist):			# enough values in the array
			for i in range(0, imax):
				if (i < step):
					self.dyvalues.append((self.yvalues[i+step] - self.yvalues[0]) / \
					(self.xvalues[i+step] - self.xvalues[0]))
				elif (i > (imax - step)):
					self.dyvalues.append((self.yvalues[imax] - self.yvalues[i-step]) / \
					(self.xvalues[imax] - self.xvalues[i-step]))
				else:
					self.dyvalues.append((self.yvalues[i+step] - self.yvalues[i-step]) / \
					(self.xvalues[i+step] - self.xvalues[i-step]))

	def analyze(self):
		self.derivation()
		self.temp_max = max(self.yvalues)
		self.temp_max_pos = gt_pos(self.yvalues, self.temp_max)
		self.temp_times = {}.fromkeys(TEMPERATURE_NAMES.split(','), 0.0)
		for idx in TEMPERATURE_NAMES.split(','):
			if (temperature_direction[idx] < 0):
				pos = gt_pos_rev(self.yvalues, temperature_leaded[idx])
			else:
				pos = gt_pos(self.yvalues, temperature_leaded[idx])
			if (pos == -1): pos = self.temp_max_pos
			self.temp_times[idx] = self.xvalues[pos]

		self.quality_values = {}.fromkeys(QUALITY_NAMES.split(','), 0.0)
		# Calculate all qaulity indicators
		self.quality_values['tPRE'] = self.temp_times['TLIQPRE'] - self.temp_times['TSTART']
		self.quality_values['tLIQUID'] = self.temp_times['TLIQPOST'] - self.temp_times['TLIQPRE']
		self.quality_values['tPOST'] = self.temp_times['TEND'] - self.temp_times['TLIQPOST']
		self.quality_values['TMAX'] = self.temp_max
		#self.quality_values['dTUP'] = sum(self.dyvalues[:self.temp_max_pos])/len(self.dyvalues[:self.temp_max_pos])
		self.quality_values['dTUP'] = max(self.dyvalues)
		#self.quality_values['dTDOWN'] = sum(self.dyvalues[self.temp_max_pos:])/len(self.dyvalues[self.temp_max_pos:])
		self.quality_values['dTDOWN'] = min(self.dyvalues)
		self.quality_index = {}.fromkeys(QUALITY_NAMES.split(','), 0.0)
		for idx in QUALITY_NAMES.split(','):
			self.quality_index[idx] = min( \
			quality_threshold/(quality_optmin[idx] - quality_min[idx])*(self.quality_values[idx] - quality_min[idx]), \
			quality_threshold/(quality_max[idx] - quality_optmax[idx])*(quality_max[idx] - self.quality_values[idx]))
			# saturate quality value between 0.0 and 1.0
			if (self.quality_index[idx] > 1.0): self.quality_index[idx] = 1.0
			if (self.quality_index[idx] < 0.0): self.quality_index[idx] = 0.0
		self.quality = sum(self.quality_index.values())/len(self.quality_index)

		self.update()


#------------------------------------------------------------------------------
class Log(object):
	profile = ''
	last_action = None
	fields = FIELD_NAMES.split(',')

	def __init__(self):
		self.init_plot()
		self.clear_logs()

	def clear_logs(self):
		self.raw_log = []
		map(Line.clear, self.lines)
		self.mode = ''

	def init_plot(self):
		#theta = radar_factory(len(QUALITY_NAMES.split(',')), frame='polygon')
		theta = radar_factory(6, frame='polygon')

		gs = gspec.GridSpec(2, 4, height_ratios=(3, 1))
		fig = plt.figure(figsize=(14, 10))

		ax_upper = fig.add_subplot(gs[0, :])
		ax_lower = fig.add_subplot(gs[-1, 0], projection='radar')
		plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.25, wspace=0.2)

		# setup plot pane for upper graph (temperature values)
		ax_upper.set_title('Temperature Profile', weight='bold', position=(0.5, 1.05),
			horizontalalignment='center', verticalalignment='center')
		ax_upper.set_ylabel(u'Temperature [°C]')
		ax_upper.set_xlim(0, MAX_X)
		ax_upper.set_xlabel(u'Time [s]')
		ax_upper.set_xticks(np.arange(0, MAX_X, 60))
		ax_upper.set_ylim(0, MAX_Y_temperature)

		# setup plot pane for lower graph (quality diagram)
		ax_lower.set_title('Quality', weight='bold', position=(0.5, 1.2),
			horizontalalignment='center', verticalalignment='center')

		# setup referent temperature levels
		for key, val in temperature_profile.iteritems():
			ax_upper.axhline(val, linestyle='dashed', linewidth=1, color='k')
			ax_upper.annotate(u'%0.0f°C' % (val), (MAX_X - 25.0, val), \
				bbox=dict(boxstyle="square", fc="w", ec='w'), ha='center', va='center')

		# select values to be plotted
		self.lines = [
			Line(ax_upper, 'Actual'),
			Line(ax_upper, 'Temp0'),
			Line(ax_upper, 'Temp1'),
			Line(ax_upper, 'Set', u'Setpoint'),
		#	Line(ax_upper, 'Temp2'),
		#	Line(ax_upper, 'Temp3'),
		]
		self.line_pcb = Line(ax_upper, 'PCB')

		ax_upper.legend()
		plt.draw()

		self.ax_upper = ax_upper
		self.ax_lower = ax_lower
		self.theta = theta

	def analyze_log(self, key):
		if (key == 'PCB'):
			# add line_pcb to
			self.lines = self.lines + [self.line_pcb]

			for lidx in self.lines:
				if (lidx.key == 'Actual'):
					# look up TSTART intersection in 'Actual' line
					ta50 = lidx.xvalues[gt_pos(lidx.yvalues, temperature_profile['TSTART'])]
				if (lidx.key == 'PCB'):
					# Adjust the time base
					tadj = ta50 - lidx.xvalues[gt_pos(lidx.yvalues, temperature_profile['TSTART'])]
					lidx.xvalues = map(lambda t : t + tadj, lidx.xvalues)

		for lidx in self.lines:
			lidx.analyze()
			if (lidx.key == key):
				for key, val in lidx.temp_times.iteritems():
					self.ax_upper.axvline(val, linestyle='dashed', linewidth=1, color='k')
					if (key == 'TPEAKPOST'):
						y = 50.0
					elif (key == 'TPEAK'):
						y = 75.0
					else:
						y = 25.0
					self.ax_upper.annotate(u'%s\n%0.0fs' % (key, val), (val, y), \
					bbox=dict(boxstyle="square", fc="w", ec='w'), ha='center', va='center')

				# Add max temperature annotation
				self.ax_upper.annotate(u'%0.1f°C' % (lidx.temp_max), \
					(lidx.xvalues[lidx.temp_max_pos], lidx.yvalues[lidx.temp_max_pos]), \
					(lidx.xvalues[lidx.temp_max_pos] + 10, lidx.yvalues[lidx.temp_max_pos] + 10), \
					arrowprops=dict(arrowstyle='fancy'), \
					bbox=dict(boxstyle="square", fc="w", ec='w'), ha='left', va='bottom')

				# Add quality data in spider chart
				self.ax_lower.plot(self.theta, lidx.quality_index.values(), color='r')
				self.ax_lower.fill(self.theta, lidx.quality_index.values(), facecolor='r', alpha=0.25)
				self.ax_lower.set_varlabels(lidx.quality_index.keys())

				# Add quality data as text box
				std = {'transform' : self.ax_lower.transAxes, 'ha' : 'right', 'va' : 'top'}
				self.ax_lower.text(1.8, 0.7, 'Quality :', std, weight='bold')
				self.ax_lower.text(2.1, 0.7, '%0.2f' % lidx.quality, std, weight='bold')
				y=0.5
				for l, x in {'Min': 2.7, 'opt Min': 3.1, 'Actual': 3.5, 'opt Max': 3.9, 'Max': 4.3}.iteritems():
					self.ax_lower.text(x, y+0.1, l, std, weight='bold')

				for key in QUALITY_NAMES.split(','):
					self.ax_lower.text(1.8, y, key + ' :',std)
					self.ax_lower.text(2.1, y, "%0.2f" % lidx.quality_index[key], std)
					# Add quality references as text box
					self.ax_lower.text(2.7, y, "%0.2f" % quality_min[key], std)
					self.ax_lower.text(3.1, y, "%0.2f" % quality_optmin[key], std)
					self.ax_lower.text(3.5, y, "%0.2f" % lidx.quality_values[key], std)
					self.ax_lower.text(3.9, y, "%0.2f" % quality_optmax[key], std)
					self.ax_lower.text(4.3, y, "%0.2f" % quality_max[key], std)
					y -= 0.1

	def save_logfiles(self):
		print 'Saved log in %s ' % logname('csv', self.profile)
		plt.savefig(logname('png', self.profile))
		plt.savefig(logname('pdf', self.profile))

		with open(logname('csv', self.profile), 'w+') as csvout:
			writer = csv.DictWriter(csvout, FIELD_NAMES.split(','))
			writer.writeheader()
			writer.writerow({'Time': 'Selected profile    %s' % self.profile})

			for l in self.raw_log:
				writer.writerow(l)

	def parse(self, line):
		values = map(str.strip, line.split(','))
		# Convert all values to float, except the mode
		values = map(float, values[0:-1]) + [values[-1], ]

		if len(values) != len(self.fields):
			raise ValueError('Expected %d fields, found %d' % (len(self.fields), len(values)))

		return dict(zip(self.fields, values))

	def process_pcb(self, logline):
		# 'EOF' if read from csv file
		if logline.startswith('EOF'):
			self.analyze_log('PCB')
			# update view
			plt.draw()
			self.save_logfiles()
			return

		# make a list from the line and clean leading and trailing white space
		values = map(str.strip, logline.split('\t'))

        # derive the field names from the first line of the file
		if ('Time' in values):
			self.fields = values
			return

        # number of values in the line must match number of fields
		if len(values) != len(self.fields):
			raise ValueError('Expected %d fields, found %d' % (len(self.fields), len(values)))
		log = dict(zip(self.fields, values))
		# convert datetime string into seconds - the reference date may be arbitrary
        # we reference it to the 50°C pass of the 'Actual' line
		log['Time'] = (datetime.datetime.strptime(log['Time'], "%Y-%m-%d_%H:%M:%S") - \
			datetime.datetime(1970, 1, 1)).total_seconds()
        # Time and MainValue form the final log
		log = {'Time':float(log['Time']), 'PCB':float(log['MainValue'].replace(',', '.'))}

        # update line object
		self.line_pcb.add(log)

		# update view
		plt.draw()

	def process_log(self, logline):
		# 'EOF' if read from csv file
		if logline.startswith('EOF'):
			if not addpcb:
				self.analyze_log('Actual')
				self.save_logfiles()
			return

		# ignore 'comments'
		if logline.startswith('#'):
			print logline
			return

		# parse Profile name
		if logline.startswith('Starting reflow with profile: '):
			self.profile = logline[30:].strip()
			return

		if logline.startswith('Selected profile'):
			self.profile = logline[20:].strip(" ,")
			return

		try:
			log = self.parse(logline)
		except ValueError, e:
			if len(logline) > 0:
				print '!!', logline
			return

		if 'Mode' in log:
			# clean up log before starting reflow
			if self.mode == 'STANDBY' and log['Mode'] in ('BAKE', 'REFLOW'):
				self.clear_logs()

			# save png graph an csv file when bake or reflow ends.
			if self.mode in ('BAKE', 'REFLOW') and log['Mode'] == 'STANDBY':
				if not addpcb:
					self.analyze_log('Actual')
					self.save_logfiles()

			self.mode = log['Mode']
			if log['Mode'] == 'BAKE':
				self.profile = 'Bake'

			if log['Mode'] in ('REFLOW', 'BAKE'):
				self.last_action = time()
				
			self.ax_upper.text(.05, .95, 'Profile: %s\nMode: %s ' % (self.profile, self.mode), \
				transform=self.ax_upper.transAxes, ha="left", va="top", bbox=dict(boxstyle="square", fc="w", ec='w'))

		if 'Time' in log and log['Time'] != 0.0:
			if 'Actual' not in log:
				return

			# update all lines
			map(lambda x: x.add(log), self.lines)
			self.raw_log.append(log)

		# update view
		plt.draw()

	def isdone(self):
		return (
			self.last_action is not None and
			time() - self.last_action > 5
		)

#------------------------------------------------------------------------------
def loop_all_profiles(num_profiles=6):
	log = Log()

	with get_tty() as port:
		profile = 0
		def select_profile(profile):
			port.write('stop\n')
			port.write('select profile %d\n' % profile)
			port.write('reflow\n')

		select_profile(profile)

		while True:
			logline = port.readline().strip()

			if log.isdone():
				log.last_action = None
				profile += 1
				if profile > 6:
					print 'Done.'
					sys.exit()
				select_profile(profile)

			log.process_log(logline)

#------------------------------------------------------------------------------
def logging_only():
	log = Log()

	with get_tty() as port:
		while True:
			log.process_log(port.readline().strip())

#------------------------------------------------------------------------------
def reading_only():
	log = Log()

	with open(fname) as port:
		rl = port.readline() # Throw away first line with field names
		rl = port.readline()
		while rl != '':
			log.process_log(rl.strip())
			rl = port.readline()

		log.process_log('EOF')
		port.close()

#------------------------------------------------------------------------------
def reading_with_pcb():

	log = Log()

	with open(fname) as port:
		rl = port.readline() # Throw away first line with field names
		rl = port.readline()
		while rl != '':
			log.process_log(rl.strip())
			rl = port.readline()

		log.process_log('EOF')
		port.close()

	with open(pfname) as port:
		rl = port.readline()
		while rl != '':
			log.process_pcb(rl.strip())
			rl = port.readline()

		log.process_pcb('EOF')
		port.close()

#------------------------------------------------------------------------------
if __name__ == '__main__':
	action = sys.argv[1] if len(sys.argv) > 1 else 'log'
	fname = pname = None
	addpcb = False

	if action == 'log':
		print 'Logging reflow sessions...'
		logging_only()

	elif action == 'csv':
		print 'Reading csv file...'
		if len(sys.argv) == 3:
			fname = sys.argv[2]
			reading_only()
		elif len(sys.argv) == 4:
			fname = sys.argv[2]
			pfname = sys.argv[3]
			print 'Adding PCB csv file...'
			addpcb = True
			reading_with_pcb()
		else:
			print 'Missing csv file name'

	elif action == 'test':
		print 'Looping over all profiles'
		loop_all_profiles()
	elif action == 'help':
		print 'Help text goes here'
	else:
		print 'Unknown action', action
		sys.exit(1)

sys.exit(0)
