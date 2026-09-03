# Demo run-of-show

Before: arm powered for 15 s, workspace clear, one block in the pick area under the overhead camera, room
lit as when the dataset was collected. Mac camera facing the presenter, hand about 50-80 cm away.

```bash
.venv/bin/python gesture_arm.py --clean --record demo.mp4
```
Answer `y` to `Workspace clear?`. The window goes full screen. (`c` brings the status strip back if you
want the numbers on camera; `f` leaves full screen.)

1. **Centre** - point at the camera and hold your fingertip inside the ring in the middle of the frame; it
   turns green and fills, then disappears. Box + skeleton + trail are on screen from the first frame; the
   small ring on the skeleton marks the fingertip that steers.
2. **Steer** - move the finger slowly left/right, then up/down; the arm follows inside its box. Park it
   above a block. Drop the hand out of frame for a second: the arm holds.
3. **Grab** - `pinch`, hold until the arc fills: toast GRAB, the arm descends, sucks, lifts. Steer the block
   somewhere else.
4. **Place** - `open-palm`: toast PLACE, the arm descends, releases, lifts. Keep steering.
5. **Home** - `thumbs-up`: banner ROUTINE, arm rises and parks. Re-centre your hand to get mirroring back.
6. **Handshake** - `peace`: four quick pumps where the arm is, about two seconds. Re-centre.
7. **Pick** - `point`: the block picker takes over (overhead camera, its model), picks the block, drops it,
   homes. Re-centre.
8. **Dead-man** - start another `peace`, then `fist` mid-shake: banner turns solid red FROZEN, the arm stops.
   Try `pinch`: ignored. `open-palm`: RESUME, back to mirroring.
9. `q` to quit: the arm vents and homes (if you quit while FROZEN it vents and stays put), the recording is saved.

If a gesture will not fire: look at the arc (is it charging?) and the log (`gesture rejected: ... < 0.70`
means the model is unsure - move the hand closer / better light; `no detection` means the box is not
there at all).
