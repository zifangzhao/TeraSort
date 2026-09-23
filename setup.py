"""Mark wheels platform-specific because the package contains a Windows DLL."""

from setuptools import Distribution, setup


class PlatformDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=PlatformDistribution)
