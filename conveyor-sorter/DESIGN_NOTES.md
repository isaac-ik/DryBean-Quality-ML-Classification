# Design notes: connecting the Dry Bean model to a physical sorter

This file records each problem that had to be solved to run this repository's classifier on a real machine. For each one it gives the options considered, the decision, the evidence behind it, and what is still unproven. Numbers marked **[validate]** are reproduced by `python vision/validate.py`. Numbers marked **[train]** come from `python vision/train_model.py`.

**Overall status:** the software pipeline and the mechanical design are checked in simulation. The machine has not been built, the Arduino firmware has not been compiled, and no real beans have been classified yet.

---

## 1. Splitting the work between the PC and the Arduino

**Problem.** Classifying a bean needs a camera, OpenCV and scikit-learn, which means a PC. Opening a gate at the right moment needs reliable timing. A PC with a USB camera has 50–200 ms of variable delay.

**Options.**
1. The PC times everything and sends "open gate N now".
2. The PC sends a delay in milliseconds.
3. The Arduino tracks the belt's position itself, and the PC only reports *which* bean and *when it was seen*.

**Decision: option 3.** The Arduino counts every step it sends to the belt motor, so it always knows exactly how far the belt has moved. The PC sends `D <gate> <age_ms>`, meaning this bean crossed the camera line `age_ms` ago. The Arduino converts that into a belt position and opens the gate when the bean reaches it. PC or USB delay can't add position error, because the belt position comes from step counts and the only time value used is the measured age of the camera frame.

**Still open.** Camera delay (`latency_ms`) is set to 80 ms and should be measured on the real camera. The firmware has not been compiled or run.

## 2. The repository has no saved model

**Problem.** The notebook trains models in PyCaret but saves no model file, and the conveyor needs one it can load.

**Options.** Save from PyCaret (heavy dependency, pinned to old library versions), or retrain with plain scikit-learn.

**Decision.** `train_model.py` retrains from `DryBeanDataset/Dry_Bean_Dataset.arff`: standard scaling plus `HistGradientBoostingClassifier`, which is the same family of model as the LightGBM model PyCaret ranked best. `--model lgbm` uses LightGBM itself. Accuracy is measured on a stratified 80/20 split, and the final model is then refit on all the data.

**Evidence.** Test accuracy is **92.5%** with all 16 features **[train]**, in line with the notebook's comparison. Bombay is perfect (100%); Sira is the weakest at 86.8% F1.

## 3. Measuring exactly what the model was trained on

**Problem.** The model only understands the dataset's 16 features. The conveyor has to compute them from a camera image in exactly the same way, or the model gets numbers it has never seen.

**Decision.** `bean_features.derive()` builds all 16 features from six basic measurements: area, perimeter, major and minor axis, convex area, and bounding box.

**Evidence.** Applied to the dataset's own base columns, the formulas reproduce every derived column of all **13,611 rows** with a worst relative error of **8.9 × 10⁻¹⁶**, i.e. exact **[validate]**.

**Finding.** In the data, `ShapeFactor2` is **Area / MajorAxisLength³**. The paper's text gives a different formula. A least-squares fit on the data found the exponents exactly, so the code follows the data.

## 4. A lower-resolution camera changes the measurements

**Problem.** The dataset camera saw beans at roughly 4× the resolution of the conveyor camera (dataset bean length ≈ 320 px; conveyor ≈ 60–80 px). Measuring by counting pixels then introduces errors that matter at this size:
- **Perimeter:** a pixel outline traces a staircase, which is **5.7% longer** than the true edge. The dataset's roundness reaches 0.991, which shows its perimeters are close to the true edge.
- **Area and convex area:** the ring of edge pixels is a much larger share of a small bean, which shifts solidity. Solidity varies very little within a variety, so even a small shift can change the predicted variety.

**Options compared.** Each was tested on 200 bean-like shapes: ellipses at dataset sizes, half of them with a kidney notch.
1. Count pixels (as the dataset's software did).
2. Measure the continuous outline: smooth the boundary, then take the edge as lying half a pixel outside the boundary-pixel centres.
3. As 2, plus an estimate of the dataset camera's own edge-pixel effect.

| Method | Model gives the same variety as at full resolution |
|---|---|
| 1. Pixel counting | 161 / 200 |
| 2. Continuous outline (smoothing 1.5 px) | 185 / 200 |
| 3. Continuous + dataset edge model | 181 / 200 |
| **2. Continuous outline (smoothing 0.7 px)** | **189 / 200** |

**Decision.** Option 2 with light smoothing (0.7 px). Smoothing less than that leaves the staircase in; smoothing more cuts the corners off concave (kidney-shaped) beans, which shortens the perimeter.

**Evidence.** On 400 new shapes, compared against the exact geometry: median errors are area +1.35%, perimeter +0.01%, major axis +0.44%, solidity −0.52% and roundness +1.39%. The model predicts the **same variety from the camera measurement as from the exact shape for 380 / 400 (95.0%)** **[validate]**.

**Lesson about testing.** The first version of this test drew shapes with OpenCV's `ellipse()`. That function fills every pixel the edge touches, so every shape came out about half a pixel too big, and the error landed on the measurement under test. The test now draws each shape 4× finer and averages down with a 50% threshold, the way a camera sensor integrates light.

**Still open.** Simulated ellipses are not real beans. Real outline accuracy also depends on lighting, focus and segmentation (see §6).

## 5. Converting camera pixels to dataset pixels

**Problem.** Eight of the 16 features carry units: area, perimeter, the two axes, convex area, equivalent diameter, and shape factors 1 and 2. They are in the dataset camera's pixels, and that camera's pixel size isn't published.

**Options.** Assume a pixel size (unknown, so it would be a guess). Use only the 8 unit-free features. Or calibrate against beans of a known variety.

**Decision.** Calibrate, with a fallback until that's done.
- `sorter.py --calibrate-scale VARIETY` measures about 30 beans of one known variety and sets scale = dataset median length for that variety ÷ measured median length.
- Every unit-carrying feature then scales exactly by that factor (lengths ×s, areas ×s², shape factors 1 and 2 ×1/s). That holds only because §4 measures the continuous outline; pixel counting would add a resolution-dependent term.
- Until calibration is done, the sorter uses a second model trained on the 8 unit-free features only.

**Evidence.**
- Calibration recovers a known scale of 4.0 as **3.992** from 30 simulated Dermason beans **[validate]**.
- The unit-free fallback reaches only **78.0%** **[train]**. Bombay recall drops to 36%, because Bombay is recognised mainly by size, so calibration matters.

## 6. Finding beans in the image

**Problem.** Bean outlines have to be separated cleanly from a moving belt made of hinged links.

**Decisions.**
- **Belt colour.** A test with a white belt **missed a pale bean** completely; a test with a blue belt detected all four test classes. The belt is printed in blue or green, because beans are never those colours, and a bean is "anything that isn't belt-coloured" (distance in Lab colour space from the calibrated belt colour).
- **Partial beans.** A bean cut off by the edge of the region of interest would give a wrong shape, so it's skipped. Each bean is classified once, as its centre crosses a trigger line in the middle of the image.
- **Tracking.** A small centroid tracker makes sure each bean is classified exactly once. In the synthetic test, 4 beans gave 4 detections with no duplicates.

**Still open.** Real shadows, specular highlights and beans touching each other haven't been tested. Touching beans would merge into one outline, which then fails the size or novelty check and goes to gate 3, not into the pack.

## 7. Objects that aren't beans of a known variety

**Problem.** A classifier always picks one of its 7 varieties, even for a stone, a husk or a broken bean.

**Decision.** Two checks:
1. **"Is it a bean at all?"** For each variety, the training data's spread is stored. A new object is "unknown" if it's further than 1.5× the 99.5th-percentile distance from *every* variety.
2. **Confidence.** Predictions below 70% probability are "unsure".

Both go to gate 3 for manual checking.

**Evidence.** A stone-shaped polygon is classified **unknown → gate 3** **[validate]**.

**Still open.** The 1.5 factor and the 70% threshold are starting values and need tuning on real rejects.

## 8. Seven varieties, three gates

**Problem.** The model outputs 7 varieties, but the machine has 3 gates plus the end of the belt.

**Decision.** The machine is framed as packaging quality control, matching the repository's goal:

| Result | Destination |
|---|---|
| Target variety | Rides off the end: the pack |
| Colour defect (black or spotted) | Gate 1 |
| Any other variety | Gate 2 |
| Unknown or unsure | Gate 3 |

Everything is set in `config.json`; `variety_gates` can send a specific variety to a specific gate.

**Known conflict.** Barbunya beans are naturally speckled and Bombay beans are dark maroon, so the colour-defect rules can misfire on those varieties. The README says to tune or remove those rules when sorting them.

## 9. Gate design and timing

**Problem.** A bean has to be diverted reliably, using a 9 g SG90 servo.

**Options.** A push arm that kicks the bean sideways (needs precise timing, and the SG90 is weak), an air jet (needs a compressor), or a diverter paddle that the moving belt pushes the bean along.

**Decision.** A diverter paddle that lies flush in the side wall when parked. When the gate opens, it swings 48° across the belt, and the belt drives the bean along it through an opening in the far wall into a chute. The 48° comes from geometry: the paddle tip ends 1.15 mm from the far wall, and at 50° it would have hit the wall.

**Timing problem found in simulation.** The bean paths in the Blender animation showed that a bean sliding along the paddle moves along the belt more slowly than the belt itself (by a factor of cos 48°). With the original "close 48 mm after the pivot" rule, a bean that landed on the far side would still be on the paddle when it closed. The fix:
- Keep the gate open from 22 mm before the pivot until the belt has moved 80 mm past it.
- That requires **at least ~105 mm between beans**, otherwise one gate window can catch the next bean. At 20 mm/s, that's about one bean every 5 s.

**Still open.** Servo angles depend on how the horn is mounted. Real friction between bean and paddle may differ from the simulation.

## 10. Fitting the mechanics to the printer and motor

| Constraint | Decision | Check |
|---|---|---|
| A1 mini bed is 180 mm; machine must be 350 mm long | Frame printed as two 175 mm halves joined by splice plates and tie bars | Every exported part's size checked against 180 mm |
| 28BYJ-48 has very little torque | Hinged-link belt on 10-pocket polygon drums: the hinges sit in the pockets, so the belt can't slip and needs no tensioning force | Belt links and drums checked for overlap in the assembled position |
| Belt runs 2 mm from the wall | All nuts sit in hex pockets inside the walls, so nothing sticks into the belt's path | Interference check |
| Gates, brackets, chutes and drums share a small space | Automated interference check of every part pair, with each gate both parked and deployed | It found and drove fixes for paddle-to-paddle collisions, bracket overlaps and drum misalignment; final result: no interferences |
| No supports | Every part is oriented with overhangs under 45° or short bridges | By design; not yet test-printed |

**Still open.** Belt friction has not been measured. The 28BYJ-48's torque margin and the printed tolerances (2 mm pin holes, 0.5 mm knuckle gaps) still need a test print.

## 11. Closing the gap between the dataset and the real camera

**Problem.** The 92.5% figure applies to the dataset's own photos. A different camera, different lighting, and bean batches from a different harvest will all shift the feature distributions.

**Decision.** Make it cheap to retrain on beans seen by the conveyor's own camera:
1. `sorter.py --log beans.csv --label SIRA` records the 16 features of beans of a known variety, in dataset units.
2. `train_model.py --extra beans.csv` retrains on the dataset plus those beans.

The CSV columns match the dataset, so the two combine directly. A round-trip test confirmed that labelled rows load and unlabelled ones are skipped.

## 12. What is verified and what isn't

| Item | Status |
|---|---|
| Feature formulas match the dataset | Verified exactly on 13,611 rows |
| Model accuracy on the dataset | 92.5% on a held-out split |
| Camera measurement fidelity | 95% same-variety agreement on 400 simulated shapes |
| Scale calibration | Recovers 3.992 for a true 4.0 in simulation |
| Non-bean rejection | Works on a synthetic stone shape |
| Mechanical fit | No interferences in the assembled CAD; all parts fit the printer bed |
| Gate timing | Simulated in Blender; firmware updated to match |
| Firmware | **Not compiled** (no Arduino toolchain was available) |
| Printed parts and tolerances | **Not printed** |
| Accuracy on real beans | **Not measured.** Next step: build, calibrate, log real beans, retrain, then measure accuracy on a held-out batch of real beans |
