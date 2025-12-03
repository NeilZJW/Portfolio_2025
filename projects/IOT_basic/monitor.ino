// ConstantSpeed.pde
// -*- mode: C++ -*-
//
// Shows how to run AccelStepper in the simplest,
// fixed speed mode with no accelerations
// \author  Mike McCauley (mikem@airspayce.com)
// Copyright (C) 2009 Mike McCauley
// $Id: ConstantSpeed.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $

#include <AccelStepper.h>
#include <string.h>

AccelStepper stepper1 (1,2,3); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5
AccelStepper stepper2 (1,4,5);
int i=0;
void setup()
{  
  stepper1.setMaxSpeed(1000);
  stepper1.setSpeed(100);
  stepper2.setMaxSpeed(1000);
  stepper2.setSpeed(100);
  Serial.begin(9600);  
}
String data;
void loop()
{  
  //stepper.runSpeed();
  //Serial.println ("HELLO");
  data = Serial.readString();
  if (data[0] == "A") {
    Serial.print(data);
    data = data.substring(2);
    Serial.print(data);
    float speed = data.toFloat();
    // Serial.print(speed);
    stepper1.setSpeed(speed);
    i = 1;
    while(i) {
      stepper1.runSpeed();
      char text = Serial.read();
      if (data == "0") {
        i = 0;
      }
    }
  }
  if (data[0] == "B") {
    //char command = Serial.read();
    Serial.print(data);
    data = data.substring(2);
    Serial.print(data);
    float speed = data.toFloat();
    // Serial.print(speed);
    stepper2.setSpeed(speed);
    i = 1;
    while(i) {
      stepper2.runSpeed();
      char text = Serial.read();
      if (data == "0") {
        i = 0;
      }
    }
  }
}
