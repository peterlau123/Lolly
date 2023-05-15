from conan import ConanFile
from conan.tools.cmake import cmake_layout

class LollyRecipe(ConanFile):
    settings ="os","compiler","build_type","arch"
    generators = "CMakeToolChain","CMakeDeps"

    def requirements(self):
        self.requires("folly/2022.01.31.00")
        self.requires("abseil/20230125.3")

    def build_requirements(self):
        self.tool_requires("cmake/3.22.6")

    def layout(self):
        cmake_layout(self)
