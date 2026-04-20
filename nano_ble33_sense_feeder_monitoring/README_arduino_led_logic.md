# Bird Feeder Scene Monitoring
## LED Logic

This project uses an Edge Impulse image classification model on an Arduino Nano 33 BLE Sense with an OV7675 camera.

The model predicts three classes:

- `bird`
- `squirrel`
- `empty`

To make the system more reliable, the Arduino does not react to every single frame directly.  
Instead, it applies a simple decision layer using:

- confidence thresholds
- score smoothing
- top1 vs top2 margin checking
- consecutive-frame confirmation

This helps reduce unstable one-frame mistakes.

## Device States

The final device uses these states:

- `EMPTY`
- `POSSIBLE_BIRD`
- `CONFIRMED_BIRD`
- `POSSIBLE_SQUIRREL`
- `CONFIRMED_SQUIRREL`
- `UNCERTAIN`

These are device-level states, not new model classes.

## LED Patterns

Because RGB output was not available on the board used in this project, the final feedback uses the built-in LED (`LED_BUILTIN`) with different flash patterns:

- **EMPTY** → LED off  
- **POSSIBLE_BIRD** → single short flash  
- **CONFIRMED_BIRD** → one longer flash  
- **POSSIBLE_SQUIRREL** → double flash  
- **CONFIRMED_SQUIRREL** → rapid repeated flashing  
- **UNCERTAIN** → triple flash  

## Decision Logic

### Bird
Bird detection is handled more carefully because bird images are more visually varied.  
A bird is only confirmed after repeated agreement across multiple frames.

### Squirrel
Squirrel detection is prioritized because it is the main target of the project and was easier for the model to recognize consistently.  
Strong squirrel detections can be confirmed more quickly.

### Empty
If the model strongly predicts `empty`, the system stays idle and the LED remains off.

### Uncertain
If the result is not strong enough, the system enters `UNCERTAIN` instead of forcing a wrong decision.

## Serial Monitor
The Arduino also prints useful information to the Serial Monitor, including:

- raw scores
- smoothed scores
- current state
- top1 and top2 classes
- confidence margin
- streak counters
