# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time    : 2021/1/20 21:56
# @author   :Mo
# @function :setup of char-similar
# @codes    :fix it and copy reference from https://github.com/TianWenQAQ/Kashgari/blob/master/setup.py


from near_synonym.version import __version__
from setuptools import find_packages, setup
import codecs


# Package meta-data.
NAME = 'near-synonym'
DESCRIPTION = 'near-synonym, 中文反义词/近义词(antonym/synonym)工具包.'
URL = 'https://github.com/yongzhuo/near-synonym'
EMAIL = '1903865025@qq.com'
AUTHOR = 'yongzhuo'
LICENSE = 'MIT'

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = "\n".join(reader.readlines())
with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name=NAME,
        version=__version__,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(),    # (exclude=('test')),
        # package_data={'near_synonym': ['*.*', 'data/*']},
      package_data={'near_synonym': ['*.*']},
      install_requires=install_requires,
        license=LICENSE,
        classifiers=['License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                     'Programming Language :: Python :: 3.11',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Programming Language :: Python :: Implementation :: PyPy'],
      )


if __name__ == "__main__":
    print("setup ok!")

# anaconda3创建环境
# conda remove -n py35 --all
# conda create -n py351 python=3.5

# 编译的2种方案:

# 方案一
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install

# 方案二
# python setup.py bdist_wheel --universal
# twine upload dist/* --verbose
### 需要 api-token
### 获取 api-token 需要 双因子验证, 可以用google的;
