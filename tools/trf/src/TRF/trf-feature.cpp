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


#include "trf-feature.h"

namespace trf
{
	void Seq::Set(Array<int> &aInt, Vocab *pv)
	{
		Set(aInt.GetBuffer(), aInt.GetNum(), pv);
	}
	void Seq::Set(int *pInt, int nLen, Vocab *pv)
	{
		Reset(nLen);
		for (int i = 0; i < nLen; i++) {
			x[word_layer][i] = pInt[i];
			/// set the seqbeg and seqend
			x[class_layer][i] = (pInt[i] < 0) ? pInt[i] : pv->GetClass(pInt[i]);
		}
	}
	void Seq::SetClass(Vocab *pv)
	{
		pv->GetClass(x[class_layer].GetBuf(), x[word_layer].GetBuf(), nLen);
	}
	void Seq::Random(Vocab *pv)
	{
		/* randomly set c */
		for (int i = 0; i < nLen; i++) {
			x[class_layer][i] = pv->RandClass();
		}

		/* randomly set a legal word*/
		for (int i = 0; i < nLen; i++) {
			Array<VocabID> *pXs = pv->GetWord(x[class_layer][i]);
			x[word_layer][i] = pXs->Get(rand() % pXs->GetNum());
		}
	}
	void Seq::Print()
	{
		for (int i = 0; i < x.GetRow(); i++) {
			lout.output(x[i].GetBuf(), nLen);
			lout << endl;
		}
	}
	void Seq::Print(File &file)
	{
		for (int i = 0; i < x.GetRow(); i++) {
			file.PrintArray("%d ", x[i].GetBuf(), nLen);
		}
		file.Print("\n");
	}

	void FeatStyle::Set(const char *pstyle)
	{
		m_aFields.Clean();
		m_nOrder = 0;

		Field fd;
		int nLen = strlen(pstyle);
		for (int i = 0; i < nLen;) {
			fd.c = pstyle[i];
			fd.i = (fd.c == '-') ? -1 : ((fd.c == 'w') ? word_layer : class_layer);
			i += 2;
			fd.n = atoi(pstyle + i);
			while (pstyle[i] != ']' && i < nLen) { 
				i++; 
			}
			i++;

			if (fd.n == 0) {
				lout_error("[FeatStyle] Set: illegal style : " << pstyle);
			}

			m_aFields.Add(fd);
			m_nOrder += fd.n;
		}
	}
	bool FeatStyle::GetKey(Seq &seq, int nPos, int nOrder, Array<int> &key) 
	{
		key.Clean();
		if (m_nOrder > nOrder) {
			return false;
		}

		int nCur = nPos;
		for (int nfield = 0; nfield < m_aFields.GetNum(); nfield++) {
			Field info = m_aFields[nfield];
			for (int i = 0; i < info.n; i++) {
				if (info.i >= 0) {
					key.Add(seq.x[info.i][nCur]);
				}
				nCur++;
			}
		}
		return true;
	}

	FeatTable::FeatTable(const char *pstyle /* = "" */)
	{
		m_ptrie = NULL;
		Reset(pstyle);

		m_aKey.SetNum(omp_get_max_threads());
		for (int i = 0; i < m_aKey.GetNum(); i++) {
			m_aKey[i] = new Array<int>(10);
		}
	}
	FeatTable::~FeatTable()
	{
		SAFE_DELETE(m_ptrie);
		SAFE_DEL_POINTER_ARRAY(m_aStyleInfo);
		SAFE_DEL_POINTER_ARRAY(m_aKey);
	}
	void FeatTable::Reset(const char *pstyle)
	{
		m_nMinOrder = 0;
		m_nMaxOrder = 0;
		m_nNum = 0;
		m_style = pstyle;
		SAFE_DELETE(m_ptrie);
		m_ptrie = new Trie<VocabID, int>;

		m_aStyleInfo.Clean();
		// analyse the style
		int nLen = strlen(pstyle);
		const char *pColon = strchr(pstyle, ':');
		if (!pColon) {
			m_aStyleInfo.Add() = new FeatStyle(pstyle);
		}
		else {
			int nBeg = pColon - pstyle;
			while (pstyle[nBeg] != '[') {
				nBeg--;
			}
			int nEnd = pColon - pstyle;
			while (pstyle[nEnd] != ']') {
				nEnd++;
			}
			
			String sub(pstyle + nBeg, nEnd - nBeg + 1);
			int iBeg = atoi(pstyle + nBeg + 1);
			int iEnd = atoi(pColon + 1);
			for (int i = iBeg; i <= iEnd; i++) {
				String subnew;
				subnew.Format("[%d]", i);
				String s = String(pstyle).Replace(sub, subnew);
				m_aStyleInfo.Add() = new FeatStyle(s);
			}
		}
		// max order
		m_nMinOrder = c_nMaxOrder;
		m_nMaxOrder = 0;
		for (int i = 0; i < m_aStyleInfo.GetNum(); i++) {
			m_nMinOrder = min(m_nMinOrder, m_aStyleInfo[i]->m_nOrder);
			m_nMaxOrder = max(m_nMaxOrder, m_aStyleInfo[i]->m_nOrder);
		}
	}
	void FeatTable::LoadFeat(Seq &seq)
	{
		for (int pos = 0; pos < seq.GetLen(); pos++) {
			Array<int> key;
			for (int i = 0; i < m_aStyleInfo.GetNum(); i++) {
				if ( false == m_aStyleInfo[i]->GetKey(seq, pos, seq.GetLen()-pos, key) )
					continue;

				bool bFound;

				// no position constraint
				int *pInt = m_ptrie->Insert(key.GetBuffer(), key.GetNum(), bFound);
				if (!bFound) *pInt = 0;
				*pInt += 1; // value is the count of each features

				if (m_aStyleInfo[i]->m_nOrder == m_nMaxOrder)
					continue;

				// at the head position
				if (pos == 0) {
					int begid = VocabID_seqbeg;
					Trie<VocabID, int> *sub = m_ptrie->InsertTrie(&begid, 1);
					int *pInt = sub->Insert(key.GetBuffer(), key.GetNum(), bFound);
					if (!bFound) *pInt = 0;
					*pInt += 1;
				}
				// at the tail position
				if (pos + m_aStyleInfo[i]->m_nOrder == seq.GetLen()) {
					int endid = VocabID_seqend;
					Trie<VocabID, int> *sub = m_ptrie->InsertTrie(key.GetBuffer(), key.GetNum());
					int *pInt = sub->Insert(&endid, 1, bFound);
					if (!bFound) *pInt = 0;
					*pInt += 1;
				}
			}
		}
	}
	int FeatTable::CutoffFeat()
	{
		// cutoff the feature first.
		bool bNeedCutoff = false;
		if (m_aCutoff.GetNum() > 0) {
			for (int i = 0; i < m_aCutoff.GetNum(); i++) {
				if (m_aCutoff[i] > 1) {
					bNeedCutoff = true;
					break;
				}
			}
		}

		// cutoff the features
		if (!bNeedCutoff) 
			return 0;

		Trie<VocabID, int> *pNewTrie = new Trie<VocabID, int>;

		// circle all the features
		int nCutNum = 0;
		for (int n = 1; n <= m_nMaxOrder + 1; n++) {
			int nCut = m_aCutoff[min(n - 1, m_aCutoff.GetNum() - 1)];
			/* as we add the <s> </s> into the trie, then the maximum order should be larger than m_nMaxOrder*/
			VocabID ngram[c_nMaxOrder];
			Trie<VocabID, int> *psub = NULL;
			TrieIter2<VocabID, int> iter(m_ptrie, ngram, n);
			while (psub = iter.Next()) {
				int nCount = *psub->GetData();
				if (nCount >= nCut) {
					*pNewTrie->Insert(ngram, n) = nCount;
				}
				else {
					nCutNum++;
				}
			}
		}

		SAFE_DELETE(m_ptrie);
		m_ptrie = pNewTrie;
		return nCutNum;
		
	}
	void FeatTable::IndexFeat(int begIdx)
	{
		// set the number(index) of the features
		int nid = begIdx;
		for (int n = 1; n <= m_nMaxOrder+1; n++) { 
			/* as we add the <s> </s> into the trie, then the maximum order should be larger than m_nMaxOrder*/
			VocabID ngram[c_nMaxOrder];
			Trie<VocabID, int> *psub = NULL;
			TrieIter2<VocabID, int> iter(m_ptrie, ngram, n, LHash_IncSort);
			while (psub = iter.Next()) {
				*psub->GetData() = nid++;
			}
		}

		m_nNum = nid - begIdx;
	}
	void FeatTable::Find(Array<int> &afeat, Array<int> &key, bool bBeg, bool bEnd)
	{
		int *pValue = m_ptrie->Find(key.GetBuffer(), key.GetNum());
		if (pValue) afeat.Add(*pValue);

		if (bBeg) { // at the begining
			int begid = VocabID_seqbeg;
			Trie<VocabID, int> *psub = m_ptrie->FindTrie(&begid, 1);
			if (psub) {
				pValue = psub->Find(key.GetBuffer(), key.GetNum());
				if (pValue)
					afeat.Add(*pValue);
			}
		}

		if (bEnd) { // at the end
			Trie<VocabID, int> *psub = m_ptrie->FindTrie(key.GetBuffer(), key.GetNum());
			int endid = VocabID_seqend;
			if (psub) {
				pValue = psub->Find(&endid, 1);
				if (pValue)
					afeat.Add(*pValue);
			}
		}
	}
	void FeatTable::Find(Array<int> &afeat, Seq &seq, int pos, int order)
	{
		Array<int> *pKey = m_aKey[omp_get_thread_num()];
		for (int i = 0; i < m_aStyleInfo.GetNum(); i++) {
			if (m_aStyleInfo[i]->m_nOrder == order) {

				m_aStyleInfo[i]->GetKey(seq, pos, order, *pKey);

				Find(afeat, *pKey, pos == 0, pos + order == seq.GetLen());
			}
		}
	}

	void FeatTable::Find(Array<int> &afeat, Seq &seq)
	{
		Array<int> key;
		for (int pos = 0; pos < seq.GetLen(); pos++)
		{
			for (int i = 0; i < m_aStyleInfo.GetNum(); i++) 
			{
				if (false == m_aStyleInfo[i]->GetKey(seq, pos, seq.GetLen() - pos, key) )
					continue;

				Find(afeat, key, pos == 0, pos + m_aStyleInfo[i]->m_nOrder == seq.GetLen());
			}
		}
	}
	void FeatTable::FindPosDep(Array<int> &afeat, Seq &seq, int pos)
	{
		int nLen = seq.GetLen();

		Array<int> *pkey = m_aKey[omp_get_thread_num()];
		for (int i = 0; i < m_aStyleInfo.GetNum(); i++) {
			int order = m_aStyleInfo[i]->m_nOrder;
			for (int n = max(0, pos - order + 1); n <= min(nLen - order, pos); n++) {
				lout_assert(true == m_aStyleInfo[i]->GetKey(seq, n, order, *pkey));
				Find(afeat, *pkey, n == 0, n + order == nLen);
			}
		}
	}
	void FeatTable::ReadT(File &file, PValue *pValue /* = NULL */)
	{
		char *pLine = file.GetLine();
		Reset(strtok(pLine, " \t\n"));
		sscanf(strtok(NULL, " \t\n"), "order=[%d,%d]", &m_nMinOrder, &m_nMaxOrder);
		sscanf(strtok(NULL, " \t\n"), "num=%d", &m_nNum);

		for (int i = 0; i < m_nNum; i++) {
			pLine = file.GetLine();
			strtok(pLine, " \t\n");

			int nid = atoi(strchr(strtok(NULL, " \t\n"), '=') + 1);
			PValue v = atof(strchr(strtok(NULL, " \t\n"), '=') + 1);
			char *p = strchr(strtok(NULL, ""), '=') + 1;

			Array<int> key;
			p = strtok(p, " \t\n");
			while (p) {
				key.Add(atoi(p));
				p = strtok(NULL, " \t\n");
			}
			*m_ptrie->Insert(key, key.GetNum()) = nid;
			if (pValue) {
				pValue[nid] = v;
			}
		}
	}
	void FeatTable::WriteT(File &file, PValue *pValue /* = NULL */)
	{
		file.Print("%s order=[%d,%d] num=%d\n", m_style.GetBuffer(), m_nMinOrder, m_nMaxOrder, m_nNum);

		int outnum = 0;
		for (int n = 1; n <= m_nMaxOrder+1; n++) {
			VocabID ngram[c_nMaxOrder];
			Trie<VocabID, int> *psub = NULL;
			TrieIter2<VocabID, int> iter(m_ptrie, ngram, n, LHash_IncSort);
			while (psub = iter.Next()) {
				int nid = *psub->GetData();

				file.Print("%s\t id=%d value=%f \t key=", m_style.GetBuffer(), nid, (pValue)? pValue[nid]:0);
				for (int i = 0; i < n; i++) {
					file.Print("%d ", ngram[i]);
				}
				file.Print("\n");

				outnum++;
			}
		}
		lout_assert(outnum == m_nNum);
	}
	void Feat::Reset(int nOrder, bool bClass)
	{
		/// add the styles
		SAFE_DEL_POINTER_ARRAY(m_aTable);
		String str;
		/* ngram */
		m_aTable.Add() = new FeatTable(str.Format("w[1:%d]", nOrder));
		if (bClass) 
			m_aTable.Add() = new FeatTable(str.Format("c[1:%d]", nOrder));
		/* skip ngram */
		for (int nSkipLen = 1; nSkipLen <= nOrder - 2; nSkipLen++) {
			for (int nSkipBeg = 1; nSkipBeg <= nOrder - nSkipLen - 1; nSkipBeg++) {
				m_aTable.Add() = new FeatTable(str.Format("w[%d]-[%d]w[%d]", nSkipBeg, nSkipLen, nOrder - nSkipLen - nSkipBeg));
				if (bClass) 
					m_aTable.Add() = new FeatTable(str.Format("c[%d]-[%d]c[%d]", nSkipBeg, nSkipLen, nOrder - nSkipLen - nSkipBeg));
			}
		}

		//m_aTable.Add() = new FeatTable("w[1]-[1:2]w[1]");
		
	}
	void Feat::Reset(const char *pfeatType)
	{
		lout << "[Feat] Reset: Load feat style form file = " << pfeatType << endl;

		SAFE_DEL_POINTER_ARRAY(m_aTable);
		File file(pfeatType, "rt");
		char *pLine;
		while (pLine = file.GetLine()) {
			// remove the comments
			char *p = strstr(pLine, "//");
			if (p) *p = '\0';
			
			p = strtok(pLine, " \t\n");
			if (!p)
				continue;
			FeatTable *pFeatTable = new FeatTable(p);

			// cutoff setting
			p = strtok(NULL, " \t\n");
			if (p) {
				pFeatTable->m_aCutoff.Clean();
				while (*p != '\0') {
					pFeatTable->m_aCutoff.Add(*p - '0');
					p++;
				}
			}

			m_aTable.Add(pFeatTable);
		}
	}
	int Feat::GetMaxOrder()
	{
		int nMaxOrder = 0;
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			nMaxOrder = max(nMaxOrder, m_aTable[i]->GetMaxOrder());
		}
		return nMaxOrder;
	}
	void Feat::Find(Array<int> &afeat, Seq &seq, int pos, int order)
	{
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			m_aTable[i]->Find(afeat, seq, pos, order);
		}
	}
	void Feat::Find(Array<int> &afeat, Seq &seq)
	{
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			m_aTable[i]->Find(afeat, seq);
		}
	}
	void Feat::FindClass(Array<int> &afeat, Seq &seq, int pos, int order)
	{
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			if (NULL == strchr(m_aTable[i]->GetStyle(), 'w')) {
				/* no word, which means containing only class */
				m_aTable[i]->Find(afeat, seq, pos, order);
			}
		}
	}
	void Feat::FindWord(Array<int> &afeat, Seq &seq, int pos, int order)
	{
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			if (strchr(m_aTable[i]->GetStyle(), 'w')) {
				/* containing word */
				m_aTable[i]->Find(afeat, seq, pos, order);
			}
		}
	}
	void Feat::FindPosDep(Array<int> &afeat, Seq &seq, int pos, int type /* = 0 */)
	{
		/* 
			type = 0: all features 
			type = 1: only class
			type = 2: expect class
		*/
		switch (type) {
		case 0:
			for (int i = 0; i < m_aTable.GetNum(); i++) {
				m_aTable[i]->FindPosDep(afeat, seq, pos);
			}
			break;
		case 1:
			for (int i = 0; i < m_aTable.GetNum(); i++) {
				if (NULL == strchr(m_aTable[i]->GetStyle(), 'w')) {
					m_aTable[i]->FindPosDep(afeat, seq, pos);
				}
			}
			break;
		case 2:
			for (int i = 0; i < m_aTable.GetNum(); i++) {
				if (strchr(m_aTable[i]->GetStyle(), 'w')) {
					m_aTable[i]->FindPosDep(afeat, seq, pos);
				}
			}
			break;
		default:
			lout_error("[Feat] FindPosDep: unknown type = " << type);
		}
		
	}
	void Feat::LoadFeatFromCorpus(const char *path, Vocab *pv)
	{
		File file(path, "rt");

		
		lout.Progress(file.fp, true, "[Feat] Load:");
		char *pLine;
		while (pLine = file.GetLine()) {
			Array<VocabID> aInt;

			//aInt.Add(VocabID_seqbeg);
			char *p = strtok(pLine, " \t\n");
			while (p) {
				aInt.Add(atoi(p));
				p = strtok(NULL, " \t\n");
			}
			//aInt.Add(VocabID_seqend);

			Seq seq;
			seq.Set(aInt, pv);

			for (int i = 0; i < m_aTable.GetNum(); i++) {
				m_aTable[i]->LoadFeat(seq);
			}
			lout.Progress(file);
		}

		// cutoff all features
		lout << "[Feat] Feat cutoff..." << endl;
		int nTotalCutNum = 0;
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			nTotalCutNum += m_aTable[i]->CutoffFeat();
		}
		lout << "[Feat] Feat cutoff num = " << nTotalCutNum << endl;
		// index all the features
		lout << "[Feat] Feat index..." << endl;
		m_nTotalNum = 0;
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			m_aTable[i]->IndexFeat(m_nTotalNum);
			m_nTotalNum += m_aTable[i]->GetNum();
		}

		// output
		lout << "[Feat] = {" << endl;
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			lout << "  " << m_aTable[i]->GetStyle() << ": " << m_aTable[i]->GetNum();
			lout << " order=[" << m_aTable[i]->GetMinOrder() << "," << m_aTable[i]->GetMaxOrder() << "]";
			lout << " cut="; 
			lout.output(m_aTable[i]->m_aCutoff.GetBuffer(), m_aTable[i]->m_aCutoff.GetNum(), "") << endl;
		}
		lout << "  total = " << m_nTotalNum << "\n}" << endl;
	}

	void Feat::WriteT(File &file, PValue *pValue /* = NULL */)
	{
		file.Print("feat-type=%d\n", m_aTable.GetNum());
		file.Print("feat={ ");
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			file.Print("%s  ", m_aTable[i]->GetStyle());
		}
		file.Print(" }\n");

		for (int i = 0; i < m_aTable.GetNum(); i++) {
			m_aTable[i]->WriteT(file, pValue);
		}
	}
	void Feat::ReadT(File &file, PValue *pValue /* = NULL */)
	{
		SAFE_DEL_POINTER_ARRAY(m_aTable);

		int nNum;
		fscanf(file, "feat-type=%d\n", &nNum);
		file.GetLine();
		m_aTable.SetNum(nNum);
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			m_aTable[i] = new FeatTable();
			m_aTable[i]->ReadT(file, pValue);
		}

		// output
		lout << "[Feat] = {" << endl;
		for (int i = 0; i < m_aTable.GetNum(); i++) {
			lout << "  " << m_aTable[i]->GetStyle() << ": " << m_aTable[i]->GetNum();
			lout << " order=[" << m_aTable[i]->GetMinOrder() << "," << m_aTable[i]->GetMaxOrder() << "]" << endl;
		}
		lout << "  total = " << m_nTotalNum << "\n}" << endl;
	}
}