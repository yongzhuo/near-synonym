# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: init


import os

from near_synonym.neg_antonym import NS


synonyms = NS.near_synonym
antonyms = NS.near_antonym
sim = NS.similarity

if os.environ.get("FLAG_MLM_ANTONYM") == "1":
    from near_synonym.mlm_antonym import MA
    mlm_synonyms = MA.near_synonym
    mlm_antonyms = MA.near_antonym
