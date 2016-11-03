// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2014-2015 Tsinghua University
// Author: wb.th08@gmail.com (Bin Wang), ozj@tsinghua.edu.cn (Zhijian Ou) 
//
// All h, cpp, cc, and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.


#pragma once
#include "trf-model.h"
#include "trf-corpus.h"
#include "wb-solve.h"
using namespace wb;

namespace trf
{
	/**
	 * \class
	 * \brief maximum likelihood objective function
	 */
	class MLfunc : public Func
	{
	protected:
		Model *m_pModel; ///< HRF model
		Vec<PValue> m_value; ///< save the temp value of type PValue.

		CorpusBase *m_pCorpusTrain; ///< training corpus
		CorpusBase *m_pCorpusValid; ///< valid corpus
		CorpusBase *m_pCorpusTest; ///< test corpus

		Vec<Prob> m_trainPi;  ///< the length distribution in training corpus

		Vec<double> m_vEmpiricalExp; ///< the empirical expectation
	public: 
		const char *m_pathOutputModel; ///< Write to model during iteration

	public:
		MLfunc() :m_pModel(NULL), m_pCorpusTrain(NULL), m_pCorpusValid(NULL), m_pCorpusTest(NULL) {
			m_pathOutputModel = NULL;
		};
		MLfunc(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL);
		void Reset(Model *pModel, CorpusBase *pTrain, CorpusBase *pValid = NULL, CorpusBase *pTest = NULL);
		virtual void SetParam(double *pdParams);
		void GetParam(double *pdParams);
		/// calculate the log-likelihood on corpus
		/* - if nCalNum = -1, calculate all the sequences in corpus;
		   - if nCalNum != -1, calculate the first min(nNum, curpus number) sequences.
		*/
		virtual double GetLL(CorpusBase *pCorpus, int nCalNum = -1, Vec<double> *pLL = NULL); 
		/// get the empirical expectation
		void GetEmpExp(CorpusBase *pCorpus, Vec<double> &vExp);

		virtual double GetValue();
		virtual void GetGradient(double *pdGradient);
		virtual int GetExtraValues(int t, double *pdValues);
	};
}