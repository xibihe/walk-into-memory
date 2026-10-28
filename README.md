# Walk Into Memory（走进回忆）

[中文](#中文) · [English](#english)

<a id="中文"></a>

## 中文

把一张照片变成可以在 Apple Vision Pro 里走进去的 3D 高斯（Gaussian splat）场景，并用手势和它互动。

仓库**不含任何照片或场景数据**，请用你自己的照片（或下面的合成测试场景）。

### 流程

```
照片 ──► Apple ML Sharp ──► 摆正地面 ──► 精简 ──► SPZ ──► visionOS App
         （不在本仓库）      └──────── pipeline/（python -m wim run）────────┘   app/
```

| 步骤 | 做什么 | 数量 / 大小 | 代码 |
| --- | --- | --- | --- |
| 1. 照片 → 3D 高斯 | ML Sharp 从单张照片预测高斯，真实尺度，OpenCV 相机坐标（y 向下） | 每张固定 1,179,648 个高斯 | Apple ML Sharp（**未包含**，需自行安装） |
| 2. 清理 + 人物保护 | 去掉照片画框外的高斯；人物（遮罩 + 深度范围）上的高斯不删不改；缩短沿视线方向的细长高斯；人物身后补一层薄背景 | — | `pipeline/wim/pipeline.py`、`mask.py` |
| 3. 摆正地面 | 在画面下部用 RANSAC 找地面平面，把场景摆平，量出相机离地高度（可按人物身高缩放） | — | `pipeline/wim/level.py` |
| 4. 精简 | 从附近一圈视点在 CPU 上渲染，按每个高斯对画面的贡献排序，保留 `--target` 个；之后校准画框边缘的不透明度并限制过大的高斯 | 默认 560,000 | `pipeline/wim/pipeline.py`、`raster.py` |
| 5. 导出 SPZ | 写成开源的 SPZ 压缩格式 + 一个 JSON 附带文件（相机高度、画框、风区） | 56 万个高斯约 7–8 MB | `pipeline/wim/spz.py` |
| 6. 头显渲染 | visionOS App 基于开源 MetalSplatter，每帧在 GPU 上排序并绘制全部高斯，双眼立体；所有效果在顶点着色器里逐个高斯计算 | — | `app/` |

### 手势

| 手势 | 效果 | 实现 |
| --- | --- | --- |
| 双手捏合（编辑模式） | 移动、旋转、缩放场景 | `ArtworkRenderer.swift` |
| 单次捏合 | 显示 / 隐藏场景，带浮现动画（左手立即；右手等双捏判定窗口过后） | 着色器 `applyReveal`，改编自 Spark |
| 右手 0.45 秒内捏两下 | 进入沉浸：边缘变柔并轻轻呼吸，周围 1.5 秒渐黑，然后切换到全沉浸（Full）；再捏两下立刻回到混合现实 | `ArtworkState.swift`、着色器 `applyEdgeBreath` |
| 张开手伸进场景并停约 0.3 秒 | 手附近的高斯变透明、缩小，向外向上飘散，像尘埃一样；收手后聚回 | `updateTouch`、着色器 `applyTouchDissolve` |
| 握拳约 0.3 秒 | 起风：越高、越绿的部分摆得越多，人物区域不动；1.5 秒渐强，约 14 秒后自动停 | `updateWind`、着色器 `applyWind` |

握拳不会被当作捏合。浮现风格（Spread、Magic、Unroll、Twister）和时长在控制窗口里调。
头动超过 1 厘米才重新排序，减少不必要的排序开销。

### 需要什么

- **pipeline**：Python 3.10+，numpy、SciPy、plyfile、OpenCV、PyTorch（CPU 即可）；`--auto-mask` 还需要 torchvision。
- **ML Sharp**：按 Apple `apple/ml-sharp` 仓库的说明安装。**模型权重仅限非商业研究用途**（见下方“许可”）。
- **app**：Xcode 26（visionOS 26 SDK），Apple Vision Pro（模拟器没有手部追踪）。

### 1. 照片 → 3D 高斯（ML Sharp）

```sh
sharp predict -i photos/ -o sharp_out/
```

### 2. 3D 高斯 → 场景文件（`pipeline/`）

```sh
cd pipeline
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m wim run ../sharp_out/garden.ply --photo ../photos/garden.jpg --auto-mask -o out/garden
python -m wim inspect out/garden.spz
```

输出 `out/garden.spz` 和 `out/garden.json`。人物遮罩三选一：`--mask person.png`（白色 = 人物）、`--auto-mask`（对 `--photo` 跑 Mask R-CNN）、`--no-person`（风景、室内）。
常用参数：`--target N`、`--edge hard|soft`、`--camera-height 米`、`--person-height 米`、`--still-box`（无风区）、`--water-box`（水面微光）。完整参数：`python -m wim run -h`。
精简在 CPU 上渲染，完整场景要很多分钟。

**没有照片也能试**：合成测试场景（一块地面、一面墙、一丛灌木和一个“人”，不含任何真实照片）：

```sh
python tests/make_synthetic.py demo/                       # demo/synthetic.ply + demo/synthetic_mask.png
python -m wim run demo/synthetic.ply --mask demo/synthetic_mask.png --target 30000 -o demo/synthetic
./tests/run_e2e.sh                                          # 端到端冒烟测试
```

### 3. 编译到 Apple Vision Pro（`app/`）

```sh
python3 app/Tools/generate_project.py      # 可选：重新生成 app/WalkIntoMemory.xcodeproj
open app/WalkIntoMemory.xcodeproj
```

1. 选中 **WalkIntoMemory** 目标 → *Signing & Capabilities*，选你的团队，把 bundle id（`com.example.WalkIntoMemory`）改成你自己的。
2. 在设备上编译运行，Swift 包（`spz-swift`、`ZIPFoundation`）会自动解析。
   不签名只验证编译：`xcodebuild -project app/WalkIntoMemory.xcodeproj -scheme WalkIntoMemory -destination 'generic/platform=visionOS' CODE_SIGNING_ALLOWED=NO build`
3. 放场景：编译前把 `name.spz` + `name.json` 放进 `app/WalkIntoMemory/SceneAssets/`，或在 App 里点 **Import scene (.spz + .json)** 同时选中两个文件。
4. 点 **Open space**，双手摆放场景，点 **Finish editing · lock placement**，捏一下让它浮现。

没有 JSON 附带文件时，App 也能显示任意 `.spz/.ply/.splat/.sog`，但伸手消散、风、沉浸柔边和照片位姿会关闭。

### 谁做了什么

| 部分 | 来源 |
| --- | --- |
| 照片 → 3D 高斯 | Apple **ML Sharp**（不在本仓库） |
| `pipeline/` 全部（清理、人物保护、摆正地面、精简、SPZ 写入、CPU 渲染器、测试） | 由 AI 编程助手 Grok 编写 |
| visionOS App、柔边呼吸 / 伸手消散 / 握拳起风 / 双捏沉浸的着色器和手势逻辑、1 厘米排序阈值 | 由 AI 编程助手 Grok 编写，基于 MetalSplatter 修改 |
| 浮现动画 | 改编自 World Labs 的 **Spark**（MIT） |
| 高斯渲染器 | Sean Cier 的 **MetalSplatter**（MIT），已内置并修改 |
| SPZ 格式 | Niantic Labs（MIT），Swift 版由 Sean Cier 移植 |

维护者：[@xibihe](https://github.com/xibihe)

### 许可

- 本项目代码：MIT，见 [LICENSE](LICENSE)。
- 第三方组件保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- **ML Sharp 的模型权重使用 Apple Machine Learning Research Model License，只允许非商业科研用途**，本仓库不包含、也不再分发它的代码或权重。用这些权重生成的场景可能同样受该许可约束，商用前请自行确认。

---

<a id="english"></a>

## English

Turn one photo into a 3D Gaussian-splat scene you can step into on Apple Vision Pro, and interact with it by hand.

The repository contains **no photos or scene data**. Use your own photo (or the synthetic test scene below).

### Pipeline

```
photo ──► Apple ML Sharp ──► level on floor ──► prune ──► SPZ ──► visionOS app
          (not included)     └──────── pipeline/ (python -m wim run) ────────┘   app/
```

| Step | What it does | Count / size | Code |
| --- | --- | --- | --- |
| 1. Photo → Gaussians | ML Sharp predicts Gaussians from a single photo, metric scale, OpenCV camera frame (y down) | always 1,179,648 Gaussians per photo | Apple ML Sharp (**not included**, install it yourself) |
| 2. Clean + protect the person | drop Gaussians outside the picture frame; Gaussians on the person (mask + depth band) are never pruned or edited; shorten needle-like Gaussians along the view ray; add a thin backplate behind the person | — | `pipeline/wim/pipeline.py`, `mask.py` |
| 3. Level on the floor | RANSAC floor plane in the lower part of the picture, level the scene, measure the camera height (optionally rescale to a person height) | — | `pipeline/wim/level.py` |
| 4. Prune | render from a ring of nearby viewpoints on the CPU, rank every Gaussian by its contribution, keep `--target`; then calibrate the picture-edge opacity and cap oversized Gaussians | 560,000 by default | `pipeline/wim/pipeline.py`, `raster.py` |
| 5. Write SPZ | open SPZ compressed format + a JSON sidecar (camera height, picture frame, wind areas) | about 7–8 MB for 560 k Gaussians | `pipeline/wim/spz.py` |
| 6. Render on the headset | visionOS app built on the open-source MetalSplatter renderer; sorts and draws every Gaussian on the GPU each frame, in stereo; all effects run per Gaussian in the vertex shader | — | `app/` |

### Gestures

| Gesture | Effect | Implementation |
| --- | --- | --- |
| Pinch with both hands (edit mode) | move, rotate and scale the scene | `ArtworkRenderer.swift` |
| Single pinch | show / hide the scene with a reveal animation (left hand: immediate; right hand: after the double-pinch window) | shader `applyReveal`, adapted from Spark |
| Right hand, two pinches within 0.45 s | enter immersion: soft breathing edges, the surroundings fade to black over 1.5 s, then the space switches to Full immersion; two pinches again return to mixed reality at once | `ArtworkState.swift`, shader `applyEdgeBreath` |
| Reach into the scene with an open hand and hold ~0.3 s | Gaussians near the hand fade, shrink and drift outwards and up like dust, then gather back when you withdraw | `updateTouch`, shader `applyTouchDissolve` |
| Hold a fist ~0.3 s | wind: higher and greener parts sway more, the person area stays still; ramps up over 1.5 s, stops by itself after ~14 s | `updateWind`, shader `applyWind` |

A fist never counts as a pinch. Reveal style (Spread, Magic, Unroll, Twister) and duration are in the control window.
The splats are re-sorted only after the head moves more than 1 cm.

### Requirements

- **pipeline**: Python 3.10+, numpy, SciPy, plyfile, OpenCV, PyTorch (CPU is enough); torchvision for `--auto-mask`.
- **ML Sharp**: install from Apple's `apple/ml-sharp` repository. **The model weights are for non-commercial research only** (see License).
- **app**: Xcode 26 with the visionOS 26 SDK and an Apple Vision Pro (hand tracking does not work in the simulator).

### 1. Photo → Gaussians (ML Sharp)

```sh
sharp predict -i photos/ -o sharp_out/
```

### 2. Gaussians → scene file (`pipeline/`)

```sh
cd pipeline
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m wim run ../sharp_out/garden.ply --photo ../photos/garden.jpg --auto-mask -o out/garden
python -m wim inspect out/garden.spz
```

Output: `out/garden.spz` and `out/garden.json`. Person mask, pick one: `--mask person.png` (white = person), `--auto-mask` (Mask R-CNN on `--photo`), or `--no-person` (landscapes, rooms).
Useful options: `--target N`, `--edge hard|soft`, `--camera-height M`, `--person-height M`, `--still-box` (wind-free area), `--water-box` (water shimmer). All options: `python -m wim run -h`.
Pruning renders on the CPU; a full-size scene takes many minutes.

**No photo yet?** A synthetic test scene (a floor, a wall, a bush and a "person"; no real photo involved):

```sh
python tests/make_synthetic.py demo/                       # demo/synthetic.ply + demo/synthetic_mask.png
python -m wim run demo/synthetic.ply --mask demo/synthetic_mask.png --target 30000 -o demo/synthetic
./tests/run_e2e.sh                                          # end-to-end smoke test
```

#### Sidecar format

```json
{ "format": "walk-into-memory.scene", "version": 1,
  "cameraHeight": 1.52, "imageSize": [1536, 2048], "edgeFrame": [/* 24 floats */],
  "wind": { "ground": 1.52, "up": -1.0, "stillBox": [x0, x1, y0, y1], "stillDepth": 3.1,
            "waterBox": [x0, x1, y0, y1] /* optional */ } }
```

### 3. Build for Apple Vision Pro (`app/`)

```sh
python3 app/Tools/generate_project.py      # optional: regenerates app/WalkIntoMemory.xcodeproj
open app/WalkIntoMemory.xcodeproj
```

1. Select the **WalkIntoMemory** target → *Signing & Capabilities*, choose your team and change the bundle id (`com.example.WalkIntoMemory`).
2. Build and run on the device. Swift packages (`spz-swift`, `ZIPFoundation`) resolve automatically.
   Compile check without signing: `xcodebuild -project app/WalkIntoMemory.xcodeproj -scheme WalkIntoMemory -destination 'generic/platform=visionOS' CODE_SIGNING_ALLOWED=NO build`
3. Add a scene: drop `name.spz` + `name.json` into `app/WalkIntoMemory/SceneAssets/` before building, or use **Import scene (.spz + .json)** in the app and select both files.
4. Tap **Open space**, place the scene with both hands, tap **Finish editing · lock placement**, then pinch once to make it appear.

Without a sidecar the app still renders any `.spz/.ply/.splat/.sog`, but reach dissolve, wind, immersion edges and photo placement are off.

### Who made what

| Part | Source |
| --- | --- |
| Photo → Gaussians | Apple **ML Sharp** (not in this repository) |
| all of `pipeline/` (cleaning, person protection, floor levelling, pruning, SPZ writer, CPU rasterizer, tests) | written by the AI coding assistant Grok |
| visionOS app; shaders and gesture logic for breathing edges, reach dissolve, fist wind and double-pinch immersion; 1 cm re-sort threshold | written by the AI coding assistant Grok, on top of a modified MetalSplatter |
| Reveal animation | adapted from **Spark** by World Labs (MIT) |
| Gaussian renderer | **MetalSplatter** by Sean Cier (MIT), vendored and modified |
| SPZ format | Niantic Labs (MIT); Swift port by Sean Cier |

Maintainer: [@xibihe](https://github.com/xibihe)

### License

- Project code: MIT, see [LICENSE](LICENSE).
- Third-party components keep their own licences, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- **The ML Sharp model weights are under the Apple Machine Learning Research Model License, which allows non-commercial research use only.** This repository does not include or redistribute ML Sharp code or weights. Scenes generated with those weights may fall under the same terms; check before any commercial use.
