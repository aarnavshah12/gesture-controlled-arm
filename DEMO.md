# Demo run-of-show

Before: arm powered for 15 s, workspace clear, one block in the pick area under the overhead camera, room
lit as when the dataset was collected. Mac camera facing the presenter, hand about 50-80 cm away.

```bash
.venv/bin/python gesture_arm.py --clean --record demo.mp4
```
Answer `y` to `Workspace clear?`. The window goes full screen. (`c` brings the status strip back if you
want the numbers on camera; `f` leaves full screen.)

1. **Centre** - hold your open hand in the middle of the frame until the "centre your hand" line goes away.
   Box + skeleton + trail are on screen from the first frame.
2. **Mirror** - move slowly left/right, then up/down. The arm follows inside its box. Drop the hand out of
   frame for a second: the arm holds.
3. **Grip** - `pinch`, hold until the arc fills: toast GRIP, pump on.
4. **Release** - `open-palm`: toast RELEASE.
5. **Home** - `thumbs-up`: banner ROUTINE, arm rises and parks. Re-centre your hand to get mirroring back.
6. **Flourish** - `peace`: wave + nod. Re-centre.
7. **Pick** - `point`: the block picker takes over (overhead camera, its model), picks the block, drops it,
   homes. Re-centre.
8. **Dead-man** - start another `peace`, then `fist` mid-wave: banner turns solid red FROZEN, the arm stops.
   Try `pinch`: ignored. `open-palm`: RESUME, back to mirroring.
9. `q` to quit: the arm vents and homes, the recording is saved.

If a gesture will not fire: look at the arc (is it charging?) and the log (`gesture rejected: ... < 0.70`
means the model is unsure - move the hand closer / better light; `no detection` means the box is not
there at all).
