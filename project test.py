import pytest

import sample2

trainingpath="C:/Users/Shiva SRK/Documents/train/*.txt"
testingpath="C:/Users/Shiva SRK/Documents/test/*.txt"
totaldata=[]
def test_dotrainingextraction():
    trainingdata=sample2.dotrainingextraction(trainingpath)
    print(trainingdata)
    assert trainingdata is not None


def test_dotestingextraction():
    assert sample2.dotestingextraction(testingpath) is not None

def test_find_entity():
    trainingdata = sample2.dotrainingextraction(trainingpath)
    assert type(sample2.find_entity(trainingdata))==list

def test_get_redactednameentities():
    testdata=sample2.dotestingextraction(testingpath)
    assert type(sample2.get_redactednameentities(testdata))==list

def test_ExtractFeatures_redact_data():
    testdata = sample2.dotestingextraction(testingpath)
    testredacteddocuments=sample2.get_redactednameentities(testdata)
    assert len(sample2.ExtractFeatures_redact_data(testredacteddocuments))>0

'''
test_dotrainingextraction()
test_dotestingextraction()
test_find_entity()
test_get_redactednameentities()
test_ExtractFeatures_redact_data()
'''

