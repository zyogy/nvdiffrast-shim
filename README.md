# nvdiffrast-shim
> Repo: `nvdiffrast-shim` · Module: `raster_ops.py` · Import as `raster_ops`.

**A drop-in, permissively-licensed replacement for nvdiffrast — built on PyTorch3D.**

[nvdiffrast](https://github.com/NVlabs/nvdiffrast) is the default GPU rasterizer
for most open-source image-to-3D pipelines. It is also released under the
**NVIDIA Source Code License**, which permits non-commercial research and
evaluation only. For an otherwise MIT-licensed project, that single dependency
is enough to block commercial use of everything downstream of it.

`raster_ops` replaces the parts of nvdiffrast those pipelines actually use with
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) (BSD-3-Clause) plus
plain PyTorch. Change one import line per file and the restriction is gone.

```python
import nvdiffrast.torch as dr    # before
import raster_ops as dr          # after
```

Contexts, call signatures, and tensor layouts are all preserved — including the
`ctx`-first calling convention — so nothing else needs to change.

---

## Why not just use PyTorch3D directly?

Because PyTorch3D is not a superset of nvdiffrast. Three things it has no
equivalent for at all had to be written from scratch:

- **Mipmapped texture sampling.** PyTorch3D has no texture unit. `raster_ops`
  implements a 2×2 box-filter mip pyramid, per-pixel LOD selection, and a
  two-tap trilinear blend.
- **UV screen-space derivatives** (`rast_db` / `diff_attrs`), computed
  analytically from the barycentric gradients.
- **Wrap modes.** `F.grid_sample` cannot express `REPEAT` or `MIRRORED_REPEAT`
  and gets the seam wrong on tiled textures, so bilinear fetches use explicit
  texel gathers.

It also papers over the conventions that differ between the two libraries — axis
flips, packed face indices, barycentric channel order, and depth handling — each
of which produces a plausible-looking but wrong image if you get it backwards.

## Verified, not assumed

Every claim below was measured against nvdiffrast itself on a real mesh:

| Check | Difference vs nvdiffrast |
|---|---|
| `texture`: nearest / linear, all wrap modes | 0.0 – 2.9e-09 *(float32 epsilon)* |
| `texture`: trilinear mipmap, LOD 0 → 4.7 | 2.9e-09 – 6.9e-09 |
| `interpolate`: vertex position | 3.96e-06 |
| implied mipmap LOD | median & p99 **exactly 0.0000**; 99.88% within 0.5 level |
| full render, 3 meshes × 6 views × 8 buffers | worst mean 0.0021 |
| UV-space texture bake | mean 0.0005 |
| metallic / roughness / alpha, multi-material | worst mean 0.0014 |

Parity was measured with harnesses that import nvdiffrast as ground truth — the
library itself never does. They aren't published here yet; open an issue if
you'd like them.

The texture test includes a **sensitivity check**: it injects a ±0.5 mip-level
error and confirms the test catches it, so an all-pass result means something.

---

## Requirements

| | |
|---|---|
| PyTorch | 2.7.0+cu128 tested; recent builds should work |
| PyTorch3D | 0.7.9 tested |
| CUDA Toolkit | needed **only to build PyTorch3D**, not at runtime |

### Installing PyTorch3D

This is the awkward part, and it is worth knowing before you start: **there are
no PyTorch3D wheels for Windows**, and the published Linux wheels lag current
PyTorch releases. On Windows it must be built from source (~15–30 min).

1. Install the CUDA Toolkit matching `torch.version.cuda`. Choose *Custom* and
   **uncheck "Display Driver"** — your existing driver is already fine, and
   replacing it can break other environments. You need
   CUDA → Development → Compiler + Libraries, and Runtime → Libraries.
2. Build from a shell where `vcvars64.bat` has been called:

```
set DISTUTILS_USE_SDK=1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set TORCH_CUDA_ARCH_LIST=8.6      REM your GPU only; saves a lot of build time
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && python -m pip install . --no-build-isolation
```

On Linux the standard `pip install .` from the PyTorch3D source tree works; the
extra environment variables above are the Windows-specific part.

> MSVC 14.44 and CUDA 12.8 compile together fine without
> `-allow-unsupported-compiler`, despite MSVC being newer than CUDA 12.8's
> official support matrix.

Verify: `python -c "import pytorch3d, pytorch3d._C; print(pytorch3d.__version__)"`

---

## Install

`raster_ops.py` is a single self-contained file — it imports only `torch`,
`typing`, and (lazily) `pytorch3d`. Copy it into your project.

```bash
curl -O https://raw.githubusercontent.com/zyogy/nvdiffrast-shim/main/raster_ops.py
```

## Usage

```python
import raster_ops as dr

ctx = dr.RasterizeCudaContext()                  # accepted; does nothing
rast, rast_db = dr.rasterize(ctx, pos_clip, faces, resolution=[H, W])
attr, attr_da = dr.interpolate(v, rast, faces, rast_db=rast_db, diff_attrs='all')
color = dr.texture(tex, uv, uv_da,
                   filter_mode='linear-mipmap-linear', boundary_mode='wrap')
```

If a package that *cannot import yours* needs the shim (a site-package, say),
inject it rather than restructuring imports:

```python
import their_module
their_module.dr = raster_ops
```

---

## Coverage

| nvdiffrast | Status | Notes |
|---|---|---|
| `RasterizeCudaContext` / `RasterizeGLContext` | ✅ | No-op; PyTorch3D needs no context. nvdiffrast 0.4.0 already aliased GL→CUDA. |
| `rasterize` | ✅ | `(bary_u, bary_v, z_ndc, tri_id+1)`. `faces_per_pixel > 1` gives depth-sorted layers. |
| `interpolate` | ✅ | Perspective-correct; analytic `diff_attrs`. |
| `texture` | ✅ | nearest / linear / mipmap-nearest / mipmap-linear; wrap / clamp / mirror / zero. |
| `antialias` | ⚠️ | Pass-through — see below. |
| `texture(boundary_mode='cube')` | ❌ | Not implemented. |
| `DepthPeeler` | ❌ | Raises `NotImplementedError`; use `faces_per_pixel=K`. |

## Known differences

**`antialias` is a documented no-op.** nvdiffrast does analytic silhouette-edge
antialiasing; PyTorch3D has no equivalent (`blur_radius` / soft blending exists
to make gradients flow, not to antialias a final image). Measured on real
renders, `dr.antialias` moves the image by ~1.1e-4 mean absolute — roughly **6×
less** than the supersampling these pipelines typically already apply — so the
function returns its input unchanged. That average is frame-wide and understates
the effect on an individual edge pixel; raise `ssaa` if you need edge AA.

**No near-plane clipping.** nvdiffrast clips in homogeneous space before the
perspective divide; PyTorch3D only sees post-divide coordinates. Behind-camera
vertices are pushed past the far plane so they cannot produce geometry in front
of the camera, but a triangle straddling the camera plane will not clip
identically. Only matters if the camera can enter the mesh.

**Edge pixels differ slightly.** Two rasterizers break sub-pixel ties
differently; ~0.66% of covered pixels pick a different triangle at shared edges.
Interior pixels agree far more closely.

**Inference-focused.** `rasterize` is differentiable through PyTorch3D and the
rest are ordinary tensor ops, but the mipmap and analytic-derivative paths have
not been validated for training.

---

## Notes

Each of these was found by measurement, not by reading documentation. If you
adapt this code, don't "simplify" them:

1. **PyTorch3D must be given clip-space `w` (view depth), not NDC z.** Its
   `perspective_correct` path divides barycentrics by whatever depth coordinate
   it receives. Passing NDC z leaves a systematic barycentric error that shows
   up as texture drift — small enough to mistake for float noise.
2. **PyTorch3D culls every fragment with depth ≤ 0.** A quad at `z=0`
   rasterizes to *zero pixels*. This silently breaks the common "rasterize in UV
   space" trick, which passes `z=0, w=1` — the pipeline completes and writes a
   blank texture. Using `w` fixes this too, since `w=1` there.
3. **`rast[..., 0:2]` are the weights of vertices 0 and 1**, not 1 and 2
   (vertex 2 gets `1-u-v`). Getting this wrong *consistently* still renders
   correct images, because interpolation cancels the error — but any code
   reading the raw barycentric buffer silently gets wrong values.

---

## License

MIT. PyTorch3D is BSD-3-Clause. Neither restricts generated output.

No nvdiffrast code is included or redistributed here — this is an independent
reimplementation of the same interface.

## Acknowledgements

Written to remove the nvdiffrast dependency from a
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) image-to-3D pipeline, but it
has no dependency on that project and should suit any codebase using nvdiffrast
for rasterization and texture sampling.

Not affiliated with or endorsed by NVIDIA. "nvdiffrast" is referenced only to
identify the library this is API-compatible with.
