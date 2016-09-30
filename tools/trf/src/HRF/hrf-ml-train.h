#pragma once
#include "hrf-model.h"
#include "trf-corpus.h"
#include "wb-solve.h"
using namespace wb;

namespace hrf
{
	typedef trf::CorpusBase CorpusBase;

	/**
	 * \class
	 * \brief maximum likelihood objective function
	 */
	class MLfunc : public Func
	{
	protected:
		Model *m_pModel; ///< HRF model

		CorpusBase *m_pCorpusTrain; ///< training corpus
		CorpusBase *m_pCorpusValid; ///< valid corpus
		CorpusBase *m_pCorpusTest; ///< test corpus

		Vec<PValue> m_values;
		Vec<Prob> m_trainPi;  ///< the length distribution in training corpus

	public: 
		const char *m_pathOutputModel; ///< Write to model during iteration

	public:
		MLfunc() :m_pModel(NULL), m_pCorpusTrain(NULL), m_pCorpusValid(NULL), m_pCorpusTest(NULL) {
			m_pathOutputModel = NULL;
		};
		MLfunc(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL);
		void Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL);
		Model *GetModel() const { return m_pModel; }
		virtual void SetParam(double *pdParams);
		void GetParam(double *pdParams);
		/// calculate the log-likelihood on corpus
		/* - if nCalNum = -1, calculate all the sequences in corpus;
		   - if nCalNum != -1, calculate the first min(nNum, curpus number) sequences.
		*/
		virtual double GetLL(CorpusBase *pCorpus, int nCalNum = -1); 

		virtual double GetValue(/*double *pdParams*/);
		virtual void GetGradient(/*double *pdParams, */double *pdGradient);
		virtual int GetExtraValues(int t, /*double *pdParams, */double *pdValues);
	};
}