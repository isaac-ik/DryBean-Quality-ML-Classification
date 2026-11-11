/*
  Bean sorter conveyor controller (Arduino Uno / Nano)

  - 28BYJ-48 + ULN2003 drives the belt (non-blocking half-stepping)
  - 3x SG90 sorting gates
  - The PC detects a bean under the camera and sends which gate it belongs to.
    The Arduino counts motor steps, so it knows exactly how far the belt has
    moved, and opens the gate when that bean arrives (and closes it after).

  Serial (115200 baud, one command per line):
    D <gate> <age_ms>   bean seen under the camera <age_ms> ago -> divert at gate 1..3
                        (gate 0 = no gate, bean rides off the end)
    R <rpm>             set belt speed (1..15 rpm, default 10)
    S                   stop belt
    G                   start belt
    T <gate>            test: open gate 1..3 for one second
    P                   print status
  Replies start with "OK", "ERR" or "#" (info).

  Wiring:
    ULN2003 IN1..IN4 -> D8, D9, D10, D11   (board powered from 5 V)
    SG90 signal gate1/2/3 -> D3, D5, D6    (power servos from a separate 5 V 2 A supply,
                                            common GND with the Arduino)
*/
#include <Servo.h>

// ---- geometry (mm) - must match the printed conveyor (cad/conveyor.py) ----
const float CAMERA_X = 60.0;                       // camera centre along the belt
const float GATE_X[3] = {96.0, 204.0, 272.0};      // gate pivot positions
const float OPEN_BEFORE = 22.0;                    // open gate when bean is this far before the pivot
const float CLOSE_AFTER = 80.0;                    // close when the belt has moved this far past the pivot
                                                   // (a bean sliding along the paddle lags the belt)
const float MM_PER_REV = 10 * 12.0;                // 10-pocket drum x 12 mm link pitch
const long STEPS_PER_REV = 4096;                   // 28BYJ-48 half-step (measure yours: 4076..4096)

// ---- servo angles: adjust after mounting the horn ----
const int PARK_DEG[3] = {90, 90, 90};              // paddle flat against wall A
const int OPEN_DEG[3] = {138, 138, 138};           // paddle swung 48 deg across the belt
const uint8_t SERVO_PIN[3] = {3, 5, 6};

const uint8_t COIL[4] = {8, 9, 10, 11};
const uint8_t HALFSTEP[8][4] = {
  {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 1, 0},
  {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 1}, {1, 0, 0, 1}};
const int8_t DIRECTION = 1;                        // flip to -1 if the belt runs backwards

const float STEPS_PER_MM = STEPS_PER_REV / MM_PER_REV;

Servo gates[3];

long beltSteps = 0;                       // total steps since boot
bool running = true;
float rpm = 10.0;
unsigned long stepIntervalUs;
unsigned long lastStepUs = 0;
uint8_t phase = 0;

// pending diversions
struct Job { long openAt; long closeAt; uint8_t gate; bool used; bool opened; };
const uint8_t MAX_JOBS = 16;
Job jobs[MAX_JOBS];
int8_t gateRefs[3] = {0, 0, 0};                    // how many jobs currently hold each gate open
unsigned long testUntil[3] = {0, 0, 0};

char line[32];
uint8_t lineLen = 0;

void setSpeed(float r) {
  rpm = constrain(r, 1.0, 15.0);
  stepIntervalUs = (unsigned long)(60.0e6 / (rpm * STEPS_PER_REV));
}

float beltSpeedMmS() { return running ? rpm * MM_PER_REV / 60.0 : 0.0; }

void coilsOff() { for (uint8_t i = 0; i < 4; i++) digitalWrite(COIL[i], LOW); }

void stepOnce() {
  phase = (phase + DIRECTION + 8) % 8;
  for (uint8_t i = 0; i < 4; i++) digitalWrite(COIL[i], HALFSTEP[phase][i]);
  beltSteps++;
}

void setGate(uint8_t g, bool open) {
  gates[g].write(open ? OPEN_DEG[g] : PARK_DEG[g]);
}

void addJob(uint8_t gate, unsigned long ageMs) {
  // where is the bean right now? it was under the camera ageMs ago
  long travelled = (long)(beltSpeedMmS() * ageMs / 1000.0 * STEPS_PER_MM);
  long camAt = beltSteps - travelled;
  float gx = GATE_X[gate - 1];
  long openAt = camAt + (long)((gx - OPEN_BEFORE - CAMERA_X) * STEPS_PER_MM);
  long closeAt = camAt + (long)((gx + CLOSE_AFTER - CAMERA_X) * STEPS_PER_MM);
  for (uint8_t i = 0; i < MAX_JOBS; i++) {
    if (!jobs[i].used) {
      jobs[i].openAt = openAt; jobs[i].closeAt = closeAt;
      jobs[i].gate = gate - 1; jobs[i].opened = false; jobs[i].used = true;
      Serial.print(F("OK D ")); Serial.print(gate);
      Serial.print(F(" in ")); Serial.print((gx - CAMERA_X) / max(beltSpeedMmS(), 0.1f), 2);
      Serial.println(F(" s"));
      return;
    }
  }
  Serial.println(F("ERR queue full"));
}

void serviceJobs() {
  for (uint8_t i = 0; i < MAX_JOBS; i++) {
    Job &j = jobs[i];
    if (!j.used) continue;
    if (!j.opened && beltSteps >= j.openAt) {
      if (gateRefs[j.gate]++ == 0) setGate(j.gate, true);
      j.opened = true;
    }
    if (j.opened && beltSteps >= j.closeAt) {
      if (--gateRefs[j.gate] <= 0) { gateRefs[j.gate] = 0; setGate(j.gate, false); }
      j.used = false;
    }
  }
  for (uint8_t g = 0; g < 3; g++) {
    if (testUntil[g] && millis() > testUntil[g]) {
      testUntil[g] = 0;
      if (gateRefs[g] == 0) setGate(g, false);
    }
  }
}

void handleLine(char *s) {
  char cmd = toupper(s[0]);
  char *args = s + 1;
  if (cmd == 'D') {
    long gate = strtol(args, &args, 10);
    long age = strtol(args, &args, 10);
    if (gate == 0) { Serial.println(F("OK D 0 pass")); return; }
    if (gate < 1 || gate > 3) { Serial.println(F("ERR gate")); return; }
    addJob((uint8_t)gate, (unsigned long)max(age, 0L));
  } else if (cmd == 'R') {
    setSpeed(atof(args));
    Serial.print(F("OK R ")); Serial.print(rpm); Serial.print(F(" rpm = "));
    Serial.print(beltSpeedMmS()); Serial.println(F(" mm/s"));
  } else if (cmd == 'S') {
    running = false; coilsOff(); Serial.println(F("OK S"));
  } else if (cmd == 'G') {
    running = true; Serial.println(F("OK G"));
  } else if (cmd == 'T') {
    long g = strtol(args, NULL, 10);
    if (g < 1 || g > 3) { Serial.println(F("ERR gate")); return; }
    setGate(g - 1, true); testUntil[g - 1] = millis() + 1000; Serial.println(F("OK T"));
  } else if (cmd == 'P') {
    Serial.print(F("# steps=")); Serial.print(beltSteps);
    Serial.print(F(" running=")); Serial.print(running);
    Serial.print(F(" speed_mm_s=")); Serial.println(beltSpeedMmS());
  } else if (cmd) {
    Serial.println(F("ERR unknown"));
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) { line[lineLen] = 0; handleLine(line); lineLen = 0; }
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < 4; i++) pinMode(COIL[i], OUTPUT);
  for (uint8_t g = 0; g < 3; g++) { gates[g].attach(SERVO_PIN[g]); setGate(g, false); }
  setSpeed(rpm);
  Serial.println(F("# bean sorter ready"));
}

void loop() {
  readSerial();
  unsigned long now = micros();
  if (running && now - lastStepUs >= stepIntervalUs) {
    lastStepUs = now;
    stepOnce();
  }
  serviceJobs();
}
