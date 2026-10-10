# FRAME & INTERIOR: The Memory of Space (completed prompt)

Source (user text, verbatim): [`memory_of_space.source.md`](memory_of_space.source.md) ·
Brief: [`examples/briefs/memory_of_space.json`](../../examples/briefs/memory_of_space.json) ·
Art direction (validated, passes `spec_checks`): [`examples/artifacts/memory_of_space.art_direction.json`](../../examples/artifacts/memory_of_space.art_direction.json)

## How conflicts were resolved
The base text and the appended DNA plus the reference image disagree on look. **The reference image and the DNA
govern the look; the base text governs the concept and the exclusions.**

| base text | DNA / reference image | resolution |
|---|---|---|
| pristine white gallery, warm-white plaster, ivory floor | brutalist concrete architecture | raw board-formed concrete exhibition hall, kept clean and museum-grade |
| soft natural daylight | cold blue-white spotlights, hard contrast, thin rim, limited volumetrics | one skylight shaft + one narrow cold spot; restrained haze in the shaft only |
| matte white ceramic robot arm | large industrial robot, black chrome, brushed titanium | large black-chrome arm with brushed-titanium joints, same calm, meditative posture |
| polished silver connectors | oxidized silver | oxidized (tarnished) silver sphere connectors |
| palette white / ivory / glass / silver + one red | cold, dark, high contrast | concrete grey, black chrome, cold white, glass; the faint warm red joint stays the **only** saturated color |
| "no dark sci-fi atmosphere" | dark, high contrast | dark but documentary: no neon, no glow effects, no flares, no smoke clouds; reads as a photographed installation |
| clear water basin | mirror + liquid, wet reflective surfaces | still water in a black-edged basin + wet polished concrete floor; reflections slightly imperfect |
| (reference shows floor cables, a cabinet, an etched glass pane) | | cables and cabinet excluded per the brief; the pane becomes the open polyhedron (Memory Frame 01) |

## Completed prompt (long-form, engine-independent)
For engines with long text context (no 77-token limit):

> **A machine constructs what space remembers.** Ultra-photorealistic architectural documentation of a contemporary
> art installation, *Memory Frame 01*, in a vast brutalist exhibition hall of raw board-formed concrete. Wide 16:9
> frame, eye-level camera, 35mm architectural lens, generous negative space.
>
> Left of center stands the artwork: an asymmetrical, open, incomplete polyhedron of transparent acrylic rods and
> translucent glass tubes joined by oxidized silver spherical connectors. Several edges are deliberately missing, so
> the empty space inside the frame becomes part of the form, suspended between order and instability. Every element
> is structurally supported. One single spherical joint glows with a very faint warm red light, the only saturated
> color in the image, the point of perception from which the structure grows.
>
> Directly beneath the sculpture, a shallow, flush black-edged basin of still, clear water mirrors the transparent
> frame and the concrete architecture. The reflection is delicate and slightly imperfect, softened by a barely
> perceptible ripple, so the image is never identical to its source.
>
> From the right, a large industrial robotic arm in black chrome with brushed titanium joints reaches in with
> calm, deliberate, almost meditative precision. Its tool tip touches one connector at the moment of placing a
> rod, clearly the maker of the work, with no anthropomorphic gesture. Arm, sculpture and water align on one
> axis as a single coherent installation; the sculpture is the focal point.
>
> Light: a cold blue-white shaft from a high rectangular skylight falls onto the sculpture and the water. One
> narrow cool spotlight, hard contrast, a thin rim light tracing the arm's silhouette and the glass edges, restrained
> volumetric haze only inside the light shaft. Wet, polished concrete floor with soft mirror-like reflections.
> Physically accurate refraction in glass and acrylic, caustics on the floor, soft global illumination, delicate
> ambient occlusion, subtle contact shadows, restrained highlights, natural exposure.
>
> Palette: concrete grey, black chrome, cold white, clear glass, brushed titanium, tarnished silver, and one faint
> muted red point. Quiet, contemplative, austere, tactile, culturally sophisticated, like a work documented for an
> international contemporary art and spatial design exhibition, not a science-fiction illustration.

**Negative / exclusions:** people, extra furniture, decorative plants, cables on the floor, equipment cabinets,
clutter, excessive LEDs, neon colors, colored light other than the single red joint, lens flares, glowing
sci-fi effects, smoke clouds, floating components without support, exaggerated futuristic machinery,
anthropomorphic robot, text, illegible text, logos, watermark.

## Compiled for the current engine (`diffusers-local` / `stabilityai/sd-turbo`)
Produced by `python -m atelier compile --config configs/live_sd_turbo.json --brief examples/briefs/memory_of_space.json --art-direction examples/artifacts/memory_of_space.art_direction.json`:

```
fine-art editorial photograph, a black chrome industrial robotic arm building a transparent glass polyhedron
above a still water basin, polished chrome (black chrome), sculptural curvature, strong reflections, crisp
highlights, faint handling marks, hard key light from above through a skylight, cool light, low fill, rim light,
8:1 contrast
```
768×432 (from 1024×576), seed 20261010, 2 steps, guidance 0. Negative prompt: none (no effect at guidance 0).

**Dropped by the 75-token CLIP budget:** palette (including the red accent), composition, must-constraints, mood,
titanium and oxidized-silver materials, and the concrete hall environment. The reference image was dropped too
(text-to-image only). On this engine the essential ideas (concrete hall, single red joint, negative space) would
**not** reach the model. Use the long-form prompt with a long-context engine, or treat this as the open
"prompt budget strategy" decision in [`docs/LIMITATIONS.md`](../LIMITATIONS.md).
No image has been generated from either prompt in this repository.
