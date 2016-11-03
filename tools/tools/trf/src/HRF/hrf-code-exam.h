#ifndef _HRF_CODE_EXAM_H_
#define _HRF_CODE_EXAM_H_

#include "hrf-model.h"
#include "hrf-sa-train.h"

namespace hrf
{
	class ModelExam
	{
	public:
		Model *pm;

	public:
		ModelExam(Model *p) :pm(p) {}
		/// set parameter values
		void SetValueAll(PValue v);
		/// set parameter values randomly
		void SetValueRand();
		/// test the exact normalization
		void TestNormalization(int nLen);
		/// test feat expectation
		void TestExpectation(int nLen);
		/// test hidden expectation
		void TestHiddenExp(int nLen);
		/// test sample
		void TestSample(int nLen = -1);
	};

	class SAExam
	{
	public:
		SAfunc *pfunc;
	public:
		SAExam(SAfunc *p) : pfunc(p){}
		/// test the sa gradient
		void TestGradient();
	};
}

#endif