# Two materialities, one artwork each

Intent (user, verbatim): "이제 다음 두 개의 물성으로 작품을 만들 것이다 각각 1개씩"

Each art direction was produced by the Claude Code subagents through the atelier pipeline:
concept-researcher → concept-generator → art-director → material-specialist. It was then merged and passed schema
and semantic validation. Runs were stopped at `SPEC_VALIDATED`: no image has been generated yet. Generation happens on
the user's Mac.

| | A: "One Sheet, Folded" | B: "Seat for No One" |
|---|---|---|
| reference materiality | brushed stainless monolith, shadowless luminous ceiling, white room, concrete | machine-milled white foam and its chips, black-framed glass, robot tool head, dark reflective floor |
| art direction | `examples/artifacts/steel_light.art_direction.json` | `examples/artifacts/foam_vitrine.art_direction.json` |
| tension | furniture made for use vs. sculpture past use | a chair shaped for a body vs. foam that would crush under one |

## A: One Sheet, Folded

**Long-form prompt** (for engines without the 77-token limit):

> Fine-art editorial photograph of a single sculpture: one continuous sheet of brushed stainless steel, 8–15 mm
> thick, bent once through a wide smooth radius into a long, low inverted-U, several metres long and knee-to-waist
> high, standing alone on a troweled grey concrete floor in an otherwise empty, matte warm off-white room. It almost
> looks like a bench, but it is too long and too singular to be used: a sculpture about how light travels around one
> bend in steel.
>
> Material: linear brushed satin stainless with the grain following the bend; soft, streaked, blurred reflections,
> never mirror chrome and never grey paint. A soft bright sheen band along the crown fades into cooler mid-grey legs;
> fine bright lines on the cut edges; faint grain variation and a slightly softer burnished patch on the crown.
>
> Light: a backlit frosted ceiling grid glowing as one even plane. Sourceless, shadowless, high-key luminous ceiling
> (about 6000 K), high fill, 2:1 contrast. Only thin contact shadows where the legs meet the concrete; the floor
> carries a faint, blurred reflection.
>
> Composition: wide 16:9, 50 mm lens, eye level slightly below standing height, about 30° oblique to the arch's long
> axis so the curve and one open end both read. The arch spans the central two-thirds just below the midline, the
> glowing ceiling band runs across the top edge, concrete fills the lower third; generous negative space; floor
> seams give the metre scale. Muted palette of warm off-white, cool ceiling glow, brushed-steel grey, concrete grey
> and graphite contact shadow. Contemplative, silent, luminous and austere, suspended between use and reverence.

**Negative:** people, reflected figure or silhouette, hands or body parts, text, logo, signage, watermark, mirror
chrome, grey plastic, cafe interior, espresso machine, cups, clothing rack, furniture showroom, other furniture or
props, dramatic hard shadows, dark moody grading, neon glow, sci-fi laboratory, rust, floating object, wide-angle
distortion.

**Compiled for sd-turbo** (compiler 1.2.0, 75/75 estimated tokens, 768×432, seed 20261010):
```
fine-art editorial photograph, one low bent brushed stainless steel arch sculpture alone on concrete under a luminous
ceiling, brushed steel folded sheet (linear brushed satin stainless, grain following the bend), gently curved, strong
reflections, diffused highlights, directional brushed grain, soft key light directly overhead from the full ceiling
plane, broad and even, neutral daylight, high fill
```
Dropped for this engine: palette, composition, constraints, mood, environment detail, imperfections, negative prompt.

## B: Seat for No One

**Long-form prompt:**

> Fine-art photograph documenting an installation: a plain, archetypal four-legged chair half-released from a raw
> block of matte white foam, standing directly on a dark polished gallery floor with no plinth. The seat and front
> legs are finished and covered in fine toolpath ridges; the backrest is still fused into the uncut rectangular
> block, which shows coarse terraced roughing steps. Dry, granular, beaded foam surfaces with no gloss: foam, not
> marble, not snow, not porcelain. A thin scatter of white chips and beads lies on the floor around the base. The chair
> is empty, milled to fit a body it could never hold.
>
> At the right frame edge, cropped and out of focus: the slim matte-black steel frame of one glass panel with a
> faint powder film, and behind it a dust-coated industrial robot tool head (dust-dulled brushed-steel spindle,
> ball-end mill with clinging foam beads), present but secondary.
>
> Light: one low raking key from the left (about 4300 K, medium hardness) grazing the terraced steps so each ridge
> casts a crisp small shadow, flagged off the background; low fill, no rim, 8:1 contrast. A dark empty gallery with
> near-black walls, no windows, no bright fixtures. The floor carries one long, clean reflection of the chair and
> nothing else.
>
> Composition: medium framing, 16:9, 50 mm lens, low three-quarter view just below seat height, clearly off axis.
> The chair and block sit on the left third, with dark negative space between them and the soft glass panel at the
> right edge. Monochrome: gallery black, matte foam white, graphite steel, chalk highlights. Quiet, austere, a withheld
> invitation, suspended mid-process, documentary restraint.

**Negative:** people, person sitting in the chair, human silhouette or reflection in glass, text, wall labels,
watermark, logo, manufacturer livery on robot, marble, snow, glossy porcelain or white plastic chair, chrome or
polished metal on the foam, fog, smoke or glowing dust, sparks, lasers, glowing LEDs, humanoid robot, central glass
room enclosure, symmetrical rows of plinths, guardian lion statues, frontal symmetrical wide gallery view, classical
statue or ornate carving, teal-and-orange grading, lens flare, neon.

**Compiled for sd-turbo** (compiler 1.2.0, 74/75 estimated tokens, 768×432, seed 20261010):
```
fine-art editorial photograph, half-carved plain chair emerging from raw matte white foam block on dark reflective
gallery floor, brushed steel tool head (dust-dulled brushed steel), gently curved, soft sheen, diffused highlights,
medium key light low raking light from left grazing the terraced foam steps, nearly parallel to the block faces,
flagged off the background
```
Dropped for this engine: palette (monochrome), composition (left third, low angle), the glass frame, constraints,
mood, negative prompt. Known risk, flagged by material-specialist: steel wording near the chair may put sheen on the
foam. If critiques show that, the first fix is to switch `materials[0]` to `other_metal`.

## Generate on the Mac
```bash
git pull
source .venv/bin/activate
python scripts/generate_once.py --config configs/live_sd_turbo_mac.json --brief examples/briefs/steel_light.json --art-direction examples/artifacts/steel_light.art_direction.json --count 4 --name steel-light-try1
python scripts/generate_once.py --config configs/live_sd_turbo_mac.json --brief examples/briefs/foam_vitrine.json --art-direction examples/artifacts/foam_vitrine.art_direction.json --count 4 --name foam-vitrine-try1
```
