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
#include "trf-vocab.h"
#include "trf-model.h"

namespace trf
{
	/**
	 * \class
	 * \brief HRF corpus base class. The derived class support the txt and bin files
	 */
	class CorpusBase
	{
	protected:
		String m_filename;

		int m_nMinLen; ///< record the minimum length;
		int m_nMaxLen; ///< record the maximum length;
		int m_nNum; ///< record the length number;
	public:
		CorpusBase() : m_nNum(0), m_nMinLen(0), m_nMaxLen(0) {};

		/// Open file and Load the file
		virtual void Reset(const char *pfilename) = 0;
		/// get the sequence in nLine
		virtual bool GetSeq(int nLine, Array<VocabID> &aSeq) = 0;
		/// get the length count
		virtual void GetLenCount(Array<int> &aLenCount) = 0;

		/// get the seq number
		virtual int GetNum() const { return m_nNum; }
		/// get the min length
		virtual int GetMinLen() const { return m_nMinLen; }
		/// get the max length
		virtual int GetMaxLen() const { return m_nMaxLen; }
		/// get the file name
		const char* GetFileName() const { return m_filename.GetBuffer();  }
	};

	/**
	 * \class
	 * \brief corpus reading the txt file
	 */
	class CorpusTxt : public CorpusBase
	{
	protected:
		Array<Array<VocabID>*> m_aSeq;
	public:
		CorpusTxt() {};
		CorpusTxt(const char *pfilename) { Reset(pfilename); }
		~CorpusTxt();
		/// Open file and Load the file
		virtual void Reset(const char *pfilename);
		/// get the sequence in nLine
		virtual bool GetSeq(int nLine, Array<VocabID> &aSeq);
		/// get the length count
		virtual void GetLenCount(Array<int> &aLenCount);
	};

	/**
	 * \class
	 * \brief Random select sequence in corpus. Used in SGD training
	 */
	class CorpusRandSelect
	{
	protected:
		CorpusBase *m_pCorpus;

		Array<int> m_aRandIdx;
		int m_nCurIdx;
	public:
		CorpusRandSelect() :m_pCorpus(NULL) {}
		CorpusRandSelect(CorpusBase *pCorpus) { Reset(pCorpus); }
		/// Reset the class
		void Reset(CorpusBase *p);
		/// Get x
		void GetSeq(Array<VocabID> &aSeq);
		/// Get Ranodm Index
		void GetIdx(int *pIdx, int nNum);
		/// Generate the random idx.
		void RandomIdx(int nNum);
	};

	

}