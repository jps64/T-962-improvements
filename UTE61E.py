# Circumflex Designs
# UT61E Meter interface example

import serial
import time
import es51922

#Create the global serial object
ser1 = serial.Serial()



#Initiate comm with UT61E Meter
def init_comm(port):
    global ser1

    ser1 = serial.Serial(port, baudrate=19200, bytesize=serial.SEVENBITS, parity=serial.PARITY_ODD, stopbits=serial.STOPBITS_ONE, timeout=2, rtscts=False, dsrdtr=True)
    print(ser1.name)
    print ("Serial Port is Open: %d\n" % ser1.is_open)
    ser1.setRTS(0)
    ser1.setDTR(1)
    time.sleep(1)
    ser1.setDTR(0)
    ser1.reset_input_buffer()
    ser1.reset_output_buffer()

def end_comm():
    ser1.close()




def get_one_reading():
    ser1.reset_input_buffer()
    ser1.reset_output_buffer()

    ser1.setDTR(1)
    #time.sleep(1)
    line = ser1.readline()
    ser1.setDTR(0)

    line = line[:12]    #keep only first 12 digits
    result = es51922.parse(line)
    #val = str(result["value"]) + " " + result["unit"]

    reading = result["value"]       #Get just the numeric measurement

    return reading




# Test Program
def test():

    init_comm("Com5:")

    for i in range(5):
        rdg = get_one_reading()
        print (rdg)
        time.sleep(0.5)


    end_comm()

    print("Done\n")



#test()



