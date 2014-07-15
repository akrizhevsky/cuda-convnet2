/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include "../include/util.cuh"
#include "../include/worker.cuh"
#include "../include/timer.cuh"

using namespace std;

/* 
 * ====================
 * WorkResult
 * ====================
 */
WorkResult::WorkResult(WorkResult::RESULTS resultType, Cost& results) : _resultType(resultType), _results(&results) {
}

WorkResult::WorkResult(WorkResult::RESULTS resultType) : _resultType(resultType), _results(NULL) {
}

WorkResult::~WorkResult() {
    delete _results; // delete NULL is ok
}

Cost& WorkResult::getResults() const {
    return *_results;
}

WorkResult::RESULTS WorkResult::getResultType() const {
    return _resultType;
}

/* 
 * ====================
 * Worker
 * ====================
 */
Worker::Worker(ConvNet& convNet) : _convNet(&convNet) {
}

Worker::~Worker() {
}

/* 
 * ====================
 * DataWorker
 * ====================
 */
DataWorker::DataWorker(ConvNet& convNet, CPUData& data) : Worker(convNet), _data(&data) {
    _dp = &convNet.getDataProvider();
    assert(_data != NULL);
    _dp->setData(*_data);
}

DataWorker::~DataWorker() {
    _dp->clearData();
}

/* 
 * ====================
 * TrainingWorker
 * ====================
 */
TrainingWorker::TrainingWorker(ConvNet& convNet, CPUData& data, double progress, bool test)
    : DataWorker(convNet, data), _progress(progress), _test(test) {
}

bool TrainingWorker::run() {
	_convNet->setTrainingProgress(_progress);
    Cost& batchCost = *new Cost();
//    Timer t, t2;
//    t2.start();
    int numMinibatches = _dp->getNumMinibatches();
    for (int i = 0; i < numMinibatches; i++) {
        Timer miniTimer;
//        miniTimer.start();
        for (int p = 0; p < _convNet->getNumPasses(); p++) {
//            t.start();
            _convNet->fprop(i, p, _test ? PASS_TEST : PASS_TRAIN);
//            printf("Worker: mini=%d, pass=%d fprop took %.2fmsec\n", i, p, t.stop());
//            t.start();
            _convNet->getCost(batchCost);
//            printf("Worker: mini=%d, pass=%d getcost took %.2fmsec\n", i, p, t.stop());

            if (!_test) {
//                t.start();
                _convNet->bprop(p, PASS_TRAIN);
//                printf("Worker: mini=%d, pass=%d bprop took %.2fmsec\n", i, p, t.stop());
//                t.start();
                _convNet->updateWeights(p);
//                printf("Worker: mini=%d, pass=%d updateWeights took %.2fmsec\n", i, p, t.stop());
            }
        }
//        printf("Worker: minibatch %d took %.2fmsec\n", i, miniTimer.stop());
    }
//    printf("Worker: batch took %.2fmsec\n", t2.stop());
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
    return false;
}

/*
 * ====================
 * SyncWorker
 * ====================
 */
SyncWorker::SyncWorker(ConvNet& convNet) : Worker(convNet) {
}

bool SyncWorker::run() {
    _convNet->copyToCPU();
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::SYNC_DONE));
    return false;
}

/*
 * ====================
 * ExitWorker
 * ====================
 */
ExitWorker::ExitWorker(ConvNet& convNet) : Worker(convNet) {
}

bool ExitWorker::run() {
    return true;
}

/* 
 * ====================
 * GradCheckWorker
 * ====================
 */
GradCheckWorker::GradCheckWorker(ConvNet& convNet, CPUData& data) 
    : DataWorker(convNet, data) {
}

bool GradCheckWorker::run() {
    _convNet->checkGradients();
    exit(0); // eh
    return true;
//    exit(0);
}

/* 
 * ====================
 * MultiviewTestWorker
 * ====================
 */
MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews, Matrix& cpuProbs, const char* logregName) 
    : DataWorker(convNet, data), _numViews(numViews), _cpuProbs(&cpuProbs), _logregName(logregName) {
//    assert(_data->getNumCases() % _numViews == 0);
//    assert(convNet.getNumReplicas() == 1); // For now?
}

MultiviewTestWorker::MultiviewTestWorker(ConvNet& convNet, CPUData& data, int numViews) 
    : DataWorker(convNet, data), _numViews(numViews), _cpuProbs(NULL), _logregName("") {
//    assert(_data->getNumCases() % _numViews == 0);
}

MultiviewTestWorker::~MultiviewTestWorker() {
//    delete _cpuProbs;
}

CPUData& MultiviewTestWorker::getMinibatch(int v, int i) {
    int numCasesPerView = _dp->getNumCases() / _numViews;
    int miniStart = v * numCasesPerView + i * _dp->getMinibatchSize();
    int miniEnd = v * numCasesPerView + min(numCasesPerView, (i + 1) * _dp->getMinibatchSize());
    CPUData& mini = _dp->getDataSlice(miniStart, miniEnd);
    return mini;
}

bool MultiviewTestWorker::run() {
    int numCasesPerView = _dp->getNumCases() / _numViews;
    int numMiniPerView = DIVUP(numCasesPerView, _dp->getMinibatchSize());

    Cost& batchCost = *new Cost();
    for (int i = 0; i < numMiniPerView; i++) {
        for (int v = 0; v < _numViews - 1; v++) {
            for (int p = 0; p < _convNet->getNumPasses(); p++) {
                _convNet->fprop(getMinibatch(v, i), p, v == 0 ? PASS_MULTIVIEW_TEST_START : PASS_MULTIVIEW_TEST);
            }
        }
        for (int p = 0; p < _convNet->getNumPasses(); p++) {
            _convNet->fprop(getMinibatch(_numViews - 1, i), p, PASS_MULTIVIEW_TEST_END);
            _convNet->getCost(batchCost);
        }
//        if (_cpuProbs != NULL) {
//            LogregCostLayer& logregLayer = *dynamic_cast<LogregCostLayer*>(&_convNet->getLayer(_logregName, 0));
//            NVMatrix::setDeviceID(logregLayer.getDeviceID());
//            Matrix& miniProbs = _cpuProbs->sliceRows(i * _dp->getMinibatchSize(),
//                                                     min(numCasesReal, (i + 1) * _dp->getMinibatchSize()));
//            NVMatrix& acts = logregLayer.getProbsAccum();
//            NVMatrix acts_T;
//            acts.transpose(acts_T);
//            acts_T.copyToHost(miniProbs);
//
//            delete &miniProbs;
//        }
    }
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
    return false;
}

/* 
 * ====================
 * FeatureWorker
 * ====================
 */
FeatureWorker::FeatureWorker(ConvNet& convNet, CPUData& data, MatrixV& ftrs, stringv& layerNames, bool deleteFeatures)
    : DataWorker(convNet, data), _ftrs(&ftrs), _layerNames(&layerNames), _deleteFeatures(deleteFeatures) {
    assert(layerNames.size() == ftrs.size());
    for (int i = 0; i < layerNames.size(); i++) {
        assert(ftrs[i]->getNumRows() == data.getNumCases());
        assert(!ftrs[i]->isTrans());
    }
}

FeatureWorker::~FeatureWorker() {
    if (_deleteFeatures) {
        for (int i = 0; i < _ftrs->size(); i++) {
            delete _ftrs->at(i);
        }
        delete _ftrs;
    }
    delete _layerNames;
}

bool FeatureWorker::run() {
    Cost& batchCost = *new Cost();
    map<int,int> repStart; // Feature write start offsets within minibatch
    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
        for (int f = 0; f < _layerNames->size(); f++) {
            repStart[f] = 0;
        }

        for (int p = 0; p < _convNet->getNumPasses(); p++) {
            _convNet->fprop(i, p, PASS_FEATURE_GEN);
            _convNet->getCost(batchCost);
            for (int f = 0; f < _layerNames->size(); f++) {

                if (_convNet->getLayer(_layerNames->at(f), 0).getFwdActiveInputReplicaIdx(p) >= 0) {
                    Matrix& miniFtrs = _ftrs->at(f)->sliceRows(i * _dp->getMinibatchSize(),
                                                               min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));

                    for (int r = 0; r < _convNet->getLayer(_layerNames->at(f), 0).getNumReplicas(); ++r) {
                        Layer& ftrLayer = _convNet->getLayer(_layerNames->at(f), r);
                        int d = ftrLayer.getDeviceID();
                        NVMatrix::setDeviceID(d);
                        NVMatrix& acts = ftrLayer.getActs();

                        Matrix& repMiniFtrs = miniFtrs.sliceRows(repStart[f],
                                                                 min(int(miniFtrs.getNumRows()), repStart[f] + acts.getLeadingDim()));

                        NVMatrix acts_T;
                        acts.transpose(false);
                        acts.transpose(acts_T);
                        acts_T.copyToHost(repMiniFtrs);
                        NVMatrix::syncStream(); // eh why not

                        delete &repMiniFtrs;

                        repStart[f] += acts.getLeadingDim();
                    }
                    delete &miniFtrs;
                }
            }
        }
    }
    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
    return false;
}

/* 
 * ====================
 * DataGradWorker
 * ====================
 */
DataGradWorker::DataGradWorker(ConvNet& convNet, CPUData& data, Matrix& dataGrads, int dataLayerIdx, int softmaxLayerIdx)
    : DataWorker(convNet, data), _dataGrads(&dataGrads), _dataLayerIdx(dataLayerIdx), _softmaxLayerIdx(softmaxLayerIdx) {
//    assert(dataGrads.getNumRows() == data.getNumCases());
//    assert(!dataGrads.isTrans());
}

DataGradWorker::~DataGradWorker() {
//    delete _dataGrads;
}

bool DataGradWorker::run() {
//    DataLayer& dataLayer = *dynamic_cast<DataLayer*>(&_convNet->getLayer(_dataLayerIdx));
//    SoftmaxLayer& softmaxLayer = *dynamic_cast<SoftmaxLayer*>(&_convNet->getLayer(_softmaxLayerIdx));
//    softmaxLayer.setDoLogregGrad(false);
//    Cost& batchCost = *new Cost(0);
//    for (int i = 0; i < _dp->getNumMinibatches(); i++) {
//        _convNet->fprop(i, PASS_TEST);
//        _convNet->getCost(batchCost);
//        softmaxLayer.getActs().apply(NVMatrixOps::Log(), softmaxLayer.getActsGrad());
//        
//        softmaxLayer.getActsGrad().addScalar(1);
//        softmaxLayer.getActsGrad().scale(-1);
//        softmaxLayer.incRcvdBInputs();
//        softmaxLayer.bprop(PASS_TEST);
//        
//        Matrix& miniDataGrads = _dataGrads->sliceRows(i * _dp->getMinibatchSize(),
//                                                      min(_dp->getNumCases(), (i + 1) * _dp->getMinibatchSize()));
//        NVMatrix& grads = dataLayer.getActsGrad();
//        NVMatrix grads_T;
//        if (grads.isTrans()) {
//            NVMatrix& soft_T = grads.getTranspose();
//            soft_T.transpose(grads_T);
//            delete &soft_T;
//        } else {
//            grads.transpose(grads_T);
//        }
//        grads_T.copyToHost(miniDataGrads);
//        delete &miniDataGrads;
//        
//        _convNet->reset();
//    }
//    cudaThreadSynchronize();
//    _convNet->getResultQueue().enqueue(new WorkResult(WorkResult::BATCH_DONE, batchCost));
    return false;
}
