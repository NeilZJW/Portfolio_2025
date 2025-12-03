#include <AccelStepper.h>

// Define stepper motor connections and motor interface type. Motor interface type must be set to 1 when using a driver:
#define dirPin 2
#define stepPin 3
#define motorInterfaceType 1

// Create a new instance of the AccelStepper class:
AccelStepper stepper = AccelStepper(motorInterfaceType, stepPin, dirPin);

void setup()
{ 
  stepper.setMaxSpeed(100);
  stepper.setAcceleration(100);
  stepper.setSpeed(100);
  serial.begin(9600);
}
char text
void loop()
{    
  // Set the target position:
  stepper.moveTo(400);
  // Run to target position with set speed and acceleration/deceleration:
  stepper.runToPosition();
  delay(1000);
  if (text == "1") {
    
  } else if (text == "0") {

  }
}