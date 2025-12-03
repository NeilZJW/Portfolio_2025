// ConstantSpeed.pde
// -*- mode: C++ -*-
// Shows how to run AccelStepper in the simplest,
// fixed speed mode with no accelerations
/// \author  Mike McCauley (mikem@airspayce.com)
// Copyright (C) 2009 Mike McCauley
// $Id: ConstantSpeed.pde,v 1.1 2011/01/05 01:51:01 mikem Exp mikem $


#include <AccelStepper.h>

AccelStepper stepper1 (1,2,3); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5
AccelStepper stepper2 (1,6,7); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5

int data;
int i=0;
void setup()
{
  stepper1.setMaxSpeed(1000);
  stepper1.setSpeed(100);
  stepper2.setMaxSpeed(1000);
  stepper2.setSpeed(100);
  Serial.begin(115200);  
  Serial.setTimeout(1);
}

SIGNAL(USART_RX_vect)
  {
  i=0;
  }


void loop()
{
  while (!Serial.available());
  //Serial.print ("HELLO");
  //stepper.runSpeed();
  if (data == 0) {
    Serial.print("FORWARD");
    i=1000;
    stepper1.setSpeed(200);
    stepper2.setSpeed(-200);
    while (i)
    {
      stepper1.runSpeed();
      stepper2.runSpeed();
      Serial.print ("FORWARD");
    }
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
    }
  }
  if (data == 1) {
    Serial.print("RIGHT");
    i=1000;
    stepper1.setSpeed(-200);
    stepper2.setSpeed(-200);
    while (i)
    {
      delay (10);
      stepper1.runSpeed();
      stepper2.runSpeed();
      i--;
      Serial.print ("RIGHT");
    }
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
    }
  }
  if (data == 2) {
    Serial.print("LEFT");
    i=1000;
    stepper1.setSpeed(200);
    stepper2.setSpeed(200);
    while (i)
    {
      delay (10);
      stepper1.runSpeed();
      stepper2.runSpeed();
      i--;
      Serial.print ("LEFT");
    }
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
    }
  }
  if (data == 3) {
    Serial.print("STOP!");
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
      Serial.println("STOP!");
    }
  }
  if (data == "4") {
    Serial.print("ERROR!");
  }
}



