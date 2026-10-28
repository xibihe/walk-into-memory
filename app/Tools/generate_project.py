"""Generate WalkIntoMemory.xcodeproj and link the vendored MetalSplatter package.

Run from anywhere: python3 app/Tools/generate_project.py
"""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
name = "WalkIntoMemory"
objects = []

def ident(label):
    return hashlib.sha1((name + label).encode()).hexdigest()[:24].upper()

def add(label, body):
    key = ident(label)
    objects.append(f"{key} = {{ {body} }};")
    return key

refs, builds = [], []
sources = sorted((root / name).glob("*.swift")) + sorted((root / name).glob("*.metal"))
for source in sources:
    kind = "sourcecode.metal" if source.suffix == ".metal" else "sourcecode.swift"
    ref = add(source.name, f'isa = PBXFileReference; lastKnownFileType = {kind}; path = "{source.name}"; sourceTree = "<group>";')
    refs.append(ref)
    builds.append(add("build" + source.name, f"isa = PBXBuildFile; fileRef = {ref};"))
resource = add("resources", 'isa = PBXFileReference; lastKnownFileType = folder; path = SceneAssets; sourceTree = "<group>";')
resource_build = add("resources-build", f"isa = PBXBuildFile; fileRef = {resource};")
source_group = add("sources", f'isa = PBXGroup; children = ({",".join(refs + [resource])},); path = {name}; sourceTree = "<group>";')
product = add("product", f'isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = {name}.app; sourceTree = BUILT_PRODUCTS_DIR;')
products = add("products", f'isa = PBXGroup; children = ({product},); name = Products; sourceTree = "<group>";')
main_group = add("main", f'isa = PBXGroup; children = ({source_group},{products},); sourceTree = "<group>";')
source_phase = add("source-phase", f'isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = ({",".join(builds)},); runOnlyForDeploymentPostprocessing = 0;')
resource_phase = add("resource-phase", f'isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = ({resource_build},); runOnlyForDeploymentPostprocessing = 0;')
package = add("metal-package", 'isa = XCLocalSwiftPackageReference; relativePath = Vendor/MetalSplatter;')
metal_product = add("metal-product", f'isa = XCSwiftPackageProductDependency; package = {package}; productName = MetalSplatter;')
splat_product = add("splat-product", f'isa = XCSwiftPackageProductDependency; package = {package}; productName = SplatIO;')
metal_build = add("metal-build", f'isa = PBXBuildFile; productRef = {metal_product};')
splat_build = add("splat-build", f'isa = PBXBuildFile; productRef = {splat_product};')
framework_phase = add("framework-phase", f'isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = ({metal_build},{splat_build},); runOnlyForDeploymentPostprocessing = 0;')

def configuration(label, mode, target=False):
    settings = {
        "CLANG_ENABLE_MODULES": "YES", "SDKROOT": "xros", "XROS_DEPLOYMENT_TARGET": "26.0",
        "SWIFT_VERSION": "6.0", "SWIFT_STRICT_CONCURRENCY": "complete",
        "DEBUG_INFORMATION_FORMAT": "dwarf" if mode == "Debug" else "dwarf-with-dsym",
        "SWIFT_OPTIMIZATION_LEVEL": "-Onone" if mode == "Debug" else "-O",
        "SWIFT_ACTIVE_COMPILATION_CONDITIONS": "DEBUG" if mode == "Debug" else "",
    }
    if target:
        settings.update({
            "PRODUCT_NAME": "$(TARGET_NAME)", "PRODUCT_BUNDLE_IDENTIFIER": "com.example.WalkIntoMemory",
            "MARKETING_VERSION": "0.1.0", "CURRENT_PROJECT_VERSION": "1",
            "SUPPORTED_PLATFORMS": "xros xrsimulator", "TARGETED_DEVICE_FAMILY": "7",
            "INFOPLIST_FILE": "Info.plist", "GENERATE_INFOPLIST_FILE": "YES",
            "CODE_SIGN_STYLE": "Automatic", "DEVELOPMENT_TEAM": "", "ENABLE_PREVIEWS": "YES",
            "LD_RUNPATH_SEARCH_PATHS": "$(inherited) @executable_path/Frameworks",
        })
    text = " ".join(f'{key} = "{value}";' for key, value in settings.items())
    return add(label + mode, f"isa = XCBuildConfiguration; buildSettings = {{ {text} }}; name = {mode};")

project_configs = [configuration("project", mode) for mode in ["Debug", "Release"]]
target_configs = [configuration("target", mode, True) for mode in ["Debug", "Release"]]
project_list = add("project-list", f'isa = XCConfigurationList; buildConfigurations = ({",".join(project_configs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target_list = add("target-list", f'isa = XCConfigurationList; buildConfigurations = ({",".join(target_configs)},); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release;')
target = add("target", f'isa = PBXNativeTarget; buildConfigurationList = {target_list}; buildPhases = ({source_phase},{framework_phase},{resource_phase},); buildRules = (); dependencies = (); name = {name}; productName = {name}; productReference = {product}; packageProductDependencies = ({metal_product},{splat_product},); productType = "com.apple.product-type.application";')
project = add("project", f'isa = PBXProject; packageReferences = ({package},); attributes = {{ LastUpgradeCheck = 2660; TargetAttributes = {{ {target} = {{ CreatedOnToolsVersion = 26.6; }}; }}; }}; buildConfigurationList = {project_list}; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base); mainGroup = {main_group}; productRefGroup = {products}; projectDirPath = ""; projectRoot = ""; targets = ({target},);')
folder = root / f"{name}.xcodeproj"
folder.mkdir(exist_ok=True)
(folder / "project.pbxproj").write_text('// !$*UTF8*$!\n{ archiveVersion = 1; classes = {}; objectVersion = 56; objects = {\n' + "\n".join(objects) + f'\n}}; rootObject = {project}; }}\n')
scheme = folder / "xcshareddata" / "xcschemes"
scheme.mkdir(parents=True, exist_ok=True)
reference = f'<BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="{target}" BuildableName="{name}.app" BlueprintName="{name}" ReferencedContainer="container:{name}.xcodeproj"/>'
(scheme / f"{name}.xcscheme").write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2660" version="1.3">
<BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES"><BuildActionEntries><BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">{reference}</BuildActionEntry></BuildActionEntries></BuildAction>
<LaunchAction buildConfiguration="Release" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.IDEFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" allowLocationSimulation="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></LaunchAction>
<ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0">{reference}</BuildableProductRunnable></ProfileAction>
<AnalyzeAction buildConfiguration="Debug"/><ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>''')
print(folder)
