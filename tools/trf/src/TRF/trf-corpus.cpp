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


#include "trf-corpus.h"

namespace trf
{
	CorpusTxt::~CorpusTxt()
	{
		// clean
		for (int i = 0; i < m_aSeq.GetNum(); i++)
			SAFE_DELETE(m_aSeq[i]);
		m_aSeq.Clean();
	}
	void CorpusTxt::Reset(const char *pfilename)
	{
		// clean
		for (int i = 0; i < m_aSeq.GetNum(); i++)
			SAFE_DELETE(m_aSeq[i]);
		m_aSeq.Clean();
		m_nMinLen = 100;
		m_nMaxLen = 0;

		// read
		File file(pfilename, "rt");
		char *pLine;
		while (pLine = file.GetLine()) {

			Array<VocabID> *pSeq = new Array<VocabID>;
			char *p = strtok(pLine, " \t\n");
			while (p) {
				pSeq->Add(atoi(p));
				p = strtok(NULL, " \t\n");
			}

			m_nMinLen = min(pSeq->GetNum(), m_nMinLen);
			m_nMaxLen = max(pSeq->GetNum(), m_nMaxLen);
			m_aSeq.Add(pSeq);
		}

		m_nNum = m_aSeq.GetNum();
	}
	bool CorpusTxt::GetSeq(int nLine, Array<VocabID> &aSeq)
	{
		if (nLine >= GetNum()) {
			return false;
		}

		aSeq.Copy(*m_aSeq[nLine]);
		return true;
	}
	void CorpusTxt::GetLenCount(Array<int> &aLenCount)
	{
		aLenCount.SetNum(m_nMaxLen + 1);
		aLenCount.Fill(0);
		for (int i = 0; i < GetNum(); i++) {
			int nLen = m_aSeq[i]->GetNum();
			aLenCount[nLen]++;
		}
	}

	

	/************************************************************************/
	/* class CorpusRandomSelect                                             */
	/************************************************************************/
	void CorpusRandSelect::Reset(CorpusBase *p)
	{
		m_pCorpus = p;
		RandomIdx(m_pCorpus->GetNum());
	}
	void CorpusRandSelect::RandomIdx(int nNum)
	{
		m_aRandIdx.SetNum(nNum);
		for (int i = 0; i < nNum; i++) {
			m_aRandIdx[i] = i;
		}
		RandomPos(m_aRandIdx, nNum, nNum);

		m_nCurIdx = 0;
	}
	void CorpusRandSelect::GetIdx(int *pIdx, int nNum)
	{
		for (int i = 0; i < nNum; i++) {
			if (m_nCurIdx >= m_pCorpus->GetNum()) {
				RandomIdx(m_pCorpus->GetNum());
			}

			pIdx[i] = m_aRandIdx[m_nCurIdx];
			m_nCurIdx++;
		}
		
	}
	void CorpusRandSelect::GetSeq(Array<VocabID> &aSeq)
	{
		if (m_nCurIdx >= m_pCorpus->GetNum()) {
			RandomIdx(m_pCorpus->GetNum());
		}

		m_pCorpus->GetSeq(m_aRandIdx[m_nCurIdx], aSeq);
		m_nCurIdx++;
	}

}