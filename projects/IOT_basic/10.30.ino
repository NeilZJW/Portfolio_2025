#include <AccelStepper.h>


AccelStepper stepper (1,2,3); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5
AccelStepper stepper1 (1,6,7); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5



void INTO_ISR(void) {


}



int i=0;
int x;
void setup() {  
  stepper.setMaxSpeed(1000);
  stepper.setSpeed(100);
  stepper1.setMaxSpeed(1000);
  stepper1.setSpeed(100);
  Serial.begin(115200);
  Serial.setTimeout(1);
  attachInterrupt(, INTO_ISR, RISING)
}






void loop() {  

  while (Serial.available()) {
    x = Serial.readString().toInt();
    if (x == 1) {
      i = 10;

      stepper.setSpeed(200);
      stepper1.setSpeed(200);
      delay (10);
      while(1){
        stepper.runSpeed();
        stepper1.runSpeed();
        Serial.print(x+1);
      }
    }
    if (x == 0) {
      delay (1);
      stepper.stop();
      stepper1.stop();
      Serial.print(x+1);
    }
  }
    
}

