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


#include "trf-vocab.h"

namespace trf
{
	Vocab::Vocab()
	{
		m_aWords.Clean();
		// 		m_aWords[VocabID_seqbeg] = Word_beg;
		// 		m_aWords[VocabID_seqend] = Word_end;
	}
	Vocab::Vocab(const char* pathVocab)
	{
		int nNum = 0;
		int nClassNum = 0;

		File file(pathVocab, "rt");
		char *pLine;
		while (pLine = file.GetLine())
		{
			VocabID id = -1;
			char *pStr = NULL; // store the word string
			char *pClass = NULL; // store the class infor

			char *p = strtok(pLine, " \t\n");
			if (!p) {
				lout_warning("[Vocab] Empty Line! (nLine=" << file.nLine << ")");
				continue;
			}

			if (strcmp(p, Word_beg) == 0) {
				lout_error("[Vocab] the input vocab exists <s>! path=" << pathVocab);
			}
			else if (strcmp(p, Word_end) == 0) {
				lout_error("[Vocab] the input vocab exists </s>! path=" << pathVocab);
			}
			else {
				id = atoi(p);
				pStr = strtok(NULL, " \t\n");
				if (String(pStr, strlen("class=")) == "class=") {
					pClass = pStr;
					pStr = NULL;
				}
				else {
					pClass = strtok(NULL, " \t\n");
				}
			}

			if (id != nNum) {
				lout_error("[Vocab] The id is not continuous (id=" << id << ")(nNum=" << nNum << ")!");
			}
			m_aWords[id] = (pStr) ? pStr : "NAN";
			m_aWordID[id] = id;


			// get the class
			if (pClass) {
				pClass += strlen("class=");
				/* read the class information */
				m_aClass[id] = atoi(pClass);
				/* count the class number */
				nClassNum = max(nClassNum, m_aClass[id] + 1);
			}

			nNum++;
		}
		
		// get the class to words
		m_aClass2Word.SetNum(nClassNum);
		m_aClass2Word.Fill(NULL);
		for (int wid = 0; wid < m_aClass.GetNum(); wid++) {
			VocabID cid = m_aClass[wid];
			if (!m_aClass2Word[cid]) {
				m_aClass2Word[cid] = new Array<int>;
			}
			m_aClass2Word[cid]->Add(wid);
		}
		for (int cid = 0; cid < m_aClass2Word.GetNum(); cid++) {
			if (m_aClass2Word[cid] == NULL) {
				lout_error("[Vocab] class " << cid << " is empty!");
			}
		}


		lout << "[Vocab] Read from " << pathVocab << endl;
		lout << "[Vocab] Read " << nNum << " words" << endl;
		lout << "[Vocab] Class = " << m_aClass2Word.GetNum() << endl;
// 		for (int cid = 0; cid < m_aClass2Word.GetNum(); cid++) {
// 			lout << "[Vocab] cid=" << cid << "\t";
// 			lout.output(m_aClass2Word[cid]->GetBuffer(), m_aClass2Word[cid]->GetNum()) << endl;
// 		}
	}

	Vocab::~Vocab()
	{
		for (int i = 0; i < m_aClass2Word.GetNum(); i++) {
			SAFE_DELETE(m_aClass2Word[i]);
		}
		m_aClass2Word.Clean();
	}

	void Vocab::GetClass(VocabID *pcid, const VocabID *pwid, int nlen)
	{
		for (int i = 0; i < nlen; i++) {
			pcid[i] = GetClass(pwid[i]);
		}
	}
}