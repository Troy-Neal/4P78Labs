#include <SPI.h>
#include "RF24.h"
#include <Motoron.h> //For motor control
#include <Deneyap_DerinlikOlcer.h> //For TOF Range Finder
/**
 * Basic template for a remote-controlled car.
 * Note that this does NOT include the guarded-control aspect,
 * because that's what YOU'RE doing!
 *
 * - The motor controller uses the I2C bus.
 * - The TOF sensor uses the same bus (on a different address).
 * - The bump sensor is on Digital 2, which also supports interrupts.
 *   include some form of automatic shutoff.
 * - D7 and D8 are the two reflectance sensors, expressed as HIGH/LOW.
 * - A1 through A3 are the three Line Follower sensors; use analogRead.
 *   (A1 is 'middle', A2 is 'left', A3 is 'right')
 * Wireless communications is handled via SPI.
 */

TofRangeFinder ranger;                      // TofRangeFinder
const int REFLECT_A = 7;
const int REFLECT_B = 8;
const int LINE_MIDDLE = A1;
const int LINE_LEFT = A2;
const int LINE_RIGHT = A3;
const int BUMP = 2;
MotoronI2C mc;
RF24 radio( 10, 9 ); //Yep, using the wireless occupies ALL our SPI pins!
uint8_t remote[]={"Remot"}; //Identifies controller
uint8_t motor[]={"Motor"}; //Identifies mobile platform

void setup() {
  Serial.begin(115200); //Needed before we start printing. Click the upper-right corner for the 'Serial Monitor'
  pinMode(REFLECT_A,INPUT);
  pinMode(REFLECT_B,INPUT);
  pinMode(LINE_MIDDLE,INPUT);
  pinMode(LINE_LEFT,INPUT);
  pinMode(LINE_RIGHT,INPUT);
  pinMode(BUMP,INPUT_PULLUP); //Note: pullup!
  
  Wire.begin(); //Needed before I2C communications
 
  //Let's get the RF controls ready!
  if (!radio.begin()) {
    //Could put something here?
    while (1) {}
  }
  radio.setPALevel(RF24_PA_LOW);  // RF24_PA_MAX is default.
  radio.setPayloadSize(3);  // We always receive 3 characters
  radio.openWritingPipe(motor);  // always uses pipe 0
  radio.openReadingPipe(1, remote);  // using pipe 1
  radio.startListening();  // put radio in RX mode
  
  //Now some boilerplate to get the Motoron controller ready:
  Wire.begin(); //Needed before I2C communications
  mc.reinitialize();
  mc.disableCrc();
  mc.clearResetFlag(); //Initializing sets reset flag, which counts as an error
  
  //Let's assume motors are allowed to stop faster than they're allowed to start
  mc.setMaxAcceleration(1, 140);
  mc.setMaxDeceleration(1, 300);
  mc.setMaxAcceleration(2, 140);
  mc.setMaxDeceleration(2, 300);
  
  ranger.begin(0x29);
}
const int16_t SPEED=800;//600;
const int16_t HALF_SPEED=400;//300;

bool flip=false;
void loop() {
  unsigned char payload[3];
  uint8_t pipe;
  if (radio.available(&pipe)) {              // is there a payload? get the pipe number that received it
    uint8_t bytes = radio.getPayloadSize();  // get the size of the payload
    radio.read(&payload, bytes);             // fetch payload from FIFO
    //payload[0] is X, [1] is Y, [2] is CZ
    //if (payload[2] == 0) digitalWrite(4,LOW); // shutoff, not working
    if (payload[2] == 0 || !digitalRead(BUMP) ){
        mc.setSpeed(1,0);
        delay(1);
        mc.setSpeed(2,0);
    } 
    else if (digitalRead(REFLECT_A) == 0 || digitalRead(REFLECT_B) == 0 || ranger.ReadDistance() < 15.0) {
        mc.setSpeed(1,HALF_SPEED);
        delay(1);
        mc.setSpeed(2,HALF_SPEED);
    }
    else if (payload[1]>150) {
      if (payload[0]<100) {
        mc.setSpeed(1,-HALF_SPEED);
        delay(1);
        mc.setSpeed(2,-SPEED);
      }
      else if (payload[0]>150) {
        mc.setSpeed(1,-SPEED);
        delay(1);
        mc.setSpeed(2,-HALF_SPEED);
      }
      else {
        mc.setSpeed(1,-SPEED);
        delay(1);
        mc.setSpeed(2,-SPEED);
      }
    }
    else if (payload[1]<100) {
      mc.setSpeed(1,SPEED);
        delay(1);
      mc.setSpeed(2,SPEED);
    }
    else {
      if (payload[0]<100) {
        mc.setSpeed(1,SPEED);
        delay(1);
        mc.setSpeed(2,-SPEED);
      }
      else if (payload[0]>150) {
        mc.setSpeed(1,-SPEED);
        delay(1);
        mc.setSpeed(2,SPEED);
      }
      else{
        mc.setSpeed(1,0);
        delay(1);
        mc.setSpeed(2,0);
      }
    }
  }
  delay(100); //Delay is necessary or Motoron will get overwhelmed You might need to increase a LITTLE
}
