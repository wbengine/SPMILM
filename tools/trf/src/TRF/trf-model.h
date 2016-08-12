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
#include "trf-feature.h"
#include "trf-alg.h"
#include <omp.h>

namespace trf
{


	class Model;

	/**
	 * \class AlgNode
	 * \brief the forward-backward algorithms for TRF model
	 */
	class AlgNode : public Algfb
	{
	private:
		Model *m_pModel;
		Seq m_seq;
	public:
		AlgNode(Model *p) :m_pModel(p){};
		virtual LogP ClusterSum(int *pSeq, int nLen, int nPos, int nOrder);
	};
	/**
	 * \class Model
	 * \brief TRF model
	 */
	class Model
	{
	public:
		Feat *m_pFeat;	 ///< hash all the features
		Vec<PValue> m_value; ///< the value for each features

		int m_maxlen; ///< the maximum length of model, excluding <s> and </s>. The min-len = 1
		Vec<Prob> m_pi; ///< the prior length distribution \pi
		Vec<LogP> m_logz; ///< the normalization constants log Z_l
		Vec<LogP> m_zeta; ///< the estimated normalization constants \zeta_l (fix \zeta_1 = 0)

		Vocab *m_pVocab;

		Mat<Prob> m_matLenJump; ///< [sample] used to propose a new length
		int m_maxSampleLen; ///< [sample] the maximum sample length, default = m_maxlen + 2

	private:
		AlgNode m_AlgNode; ///< the  forward-backward calculation each node

	public:
		/// constructor
		Model(Vocab *pv) :
			m_pFeat(NULL), 
			m_maxlen(0), 
			m_pVocab(pv), 
			m_AlgNode(this){
			m_nLenJumpAccTimes = 0;
			m_nLenJumpTotalTime = 0;
			m_nSampleHAccTimes = 0;
			m_nSampleHTotalTimes = 0;
		};
		/// constructor
		Model(Vocab *pv, int maxlen) :
			m_pFeat(NULL),
			m_maxlen(0),
			m_pVocab(pv),
			m_AlgNode(this) {
			Reset(pv, maxlen);
			m_nLenJumpAccTimes = 0;
			m_nLenJumpTotalTime = 0;
			m_nSampleHAccTimes = 0;
			m_nSampleHTotalTimes = 0;
		}
		/// destructor
		~Model()
		{
			SAFE_DELETE(m_pFeat);
		}
		/// Get max-len
		int GetMaxLen() const { return m_maxlen; }
		/// Get Vocab
		Vocab *GetVocab() const { return m_pVocab; }
		/// Get maximum order
		int GetMaxOrder() const { return m_pFeat->GetMaxOrder(); }
		/// Get parameter number
		int GetParamNum() const { return m_pFeat->GetNum(); }
		/// reset, the maxlen is the length excluding the beg/end symbols.
		void Reset(Vocab *pv, int maxlen);
		/// Set the parameters
		virtual void SetParam(PValue *pValue);
		/// Get the paremetre vector
		void GetParam(PValue *pValue);
		/// Set the pi
		void SetPi(Prob *pPi);
		/// Set updated zeta
		template <typename T>
		void SetZeta(T *pzeta)
		{
			ExactNormalize(1);
			for (int i = 1; i <= m_maxlen; i++) {
				m_zeta[i] = (LogP)( pzeta[i] - pzeta[1] );
				m_logz[i] = (LogP)( m_zeta[i] + m_logz[1] );
			}
		}
		/// calculate the probability
		LogP GetLogProb(Seq &seq, bool bNorm = true);
		/// load ngram features from corpus
		void LoadFromCorpus(const char *pcorpus, const char *pfeatstyle, int nOrder);
		/// Count the feature number in a sequence
		void FeatCount(Seq &seq, double *pCount, double dadd = 1.0);

		/// Read Model
		void ReadT(const char *pfilename);
		/// Write Model
		void WriteT(const char *pfilename);

		/************************************************************************/
		/*exactly calculation functions                                         */
		/************************************************************************/
	public:
		/// [exact] Calculate the logP in each cluster. Only used for forward-backword algorithms ( class AlgNode)
		LogP ClusterSum(Seq &seq, int nPos, int nOrder);
		/// [exact] Exact Normalization, return the logz of given length
		double ExactNormalize(int nLen);
		/// [exact] Exact Normalization
		void ExactNormalize();
		/// [exact] E_{p_l}[f]: Exactly calculate the expectation over x and h for length nLen
		void GetNodeExp(int nLen, double *pExp);
		/// [exact] sum_l { n_l/n * E_{p_l}[f] }: Exactly calculate the expectation over x and h
		void GetNodeExp(double *pExp, Prob *pLenProb = NULL);

		/************************************************************************/
		/*sampling functions                                                    */
		/************************************************************************/
	public:
		int m_nLenJumpAccTimes; ///< lenght jump the acceptance times
		int m_nLenJumpTotalTime; ///< total times of length jump
		int m_nSampleHAccTimes; ///< sample H the acceptance times
		int m_nSampleHTotalTimes; ///< sample H the total times
		/// [sample] Perform one train-dimensional mixture sampling
		void Sample(Seq &seq);
		/// [sample] Local Jump - sample a new length
		void LocalJump(Seq &seq);
		/// [sample] Markov Move - perform the gibbs sampling
		void MarkovMove(Seq &seq);
		/// [sample] Propose the length, using the variable m_matLenJump
		LogP ProposeLength(int nOld, int &nNew, bool bSample);
		/// [sample] Propose the c_{i} at position i. Then return the propose probability R(c_i|h_i,c_{other})
		LogP ProposeC0(VocabID &ci, Seq &seq, int nPos, bool bSample);
		/// [sample] Return the propose distribution of c_i at position nPos
		void ProposeCProbs(VecShell<LogP> &logps, Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample class c_i.
		/* To reduce the computation cost, we using the following function to replace GetLogProb
		   when sampling c_i at position i in function ProposeCProbs and ProposeC0.
		   There we only consinder the features depending on c_i and indenpending with w_i,
		   i.e. calculating the propose prob without knowing the w_i at position i.
		*/
		LogP GetReducedModelForC(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced model to sample word w_i.
		LogP GetReducedModelForW(Seq &seq, int nPos);
		/// [sample] A unnormalized reduced depending on nPos.
		LogP GetReducedModel(Seq &seq, int nPos);
		/// [sample] given c_i, summate the probabilities of x_i, i.e. P(c_i)
		LogP GetMarginalProbOfC(Seq &seq, int nPos);
		/// [sample] Sample the c_i at position nPos without x_i.
		/* the only differnece with ProposeC0 is than 
		   SampleC will accept the current class after propose it.
		   While ProposeC0 not.
		   ProposeC0 used in local jump. It cannot accept the propose c0 as there is no intial value of c_i.
		   SampleC used in Markov move.
		*/
		void SampleC(Seq &seq, int nPos);
		/// [sample] Sample the x_i at position nPos
		/* if bSample=ture, then sample x[nPos]. Otherwise only calculate the conditional probabilities of current x[nPos]. */
		LogP SampleX(Seq &seq, int nPos, bool bSample = true);
	public:
		/// perform AIS to calculate the normalization constants, return the logz of given length
		LogP AISNormalize(int nLen, int nChain, int nInter);
		void AISNormalize(int nChain, int nInter);
	};
}