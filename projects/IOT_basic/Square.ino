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

int i=0;
int c = 4; // control the number of turning right
void setup()
{
  stepper1.setMaxSpeed(1000);
  stepper1.setSpeed(100);
  stepper2.setMaxSpeed(1000);
  stepper2.setSpeed(100);
  Serial.begin(9600);  
}

void loop()
{
  Serial.println ("HELLO");
  while(c) {
  //stepper.runSpeed();
    i=1000;
    stepper1.setSpeed(200);
    stepper2.setSpeed(-200);
    while (i)
    {
      delay (10);
      stepper1.runSpeed();
      stepper2.runSpeed();
      i--;
      Serial.println (i);
    }
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
    }
    
    i=1000;
    stepper1.setSpeed(-200);
    stepper2.setSpeed(-200);
    while (i)
    {
      delay (10);
      stepper1.runSpeed();
      stepper2.runSpeed();
      i--;
      Serial.println (i);
    }
    i=1000;
    while (i)
    {
      delay (1);
      stepper1.stop();
      stepper2.stop();
      i--;
    }
    c--;
  }

  // i=10
  // // back
  // i=1000;
  // stepper1.setSpeed(-200);
  // stepper2.setSpeed(200);
  // while (i)
  // {
  //   delay (10);
  //   stepper1.runSpeed();
  //   stepper2.runSpeed();
  //   i--;
  // }
}

 

