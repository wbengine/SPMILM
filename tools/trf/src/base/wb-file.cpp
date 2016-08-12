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



#include "wb-file.h"

namespace wb
{
	bool File::Open(const char *path, const char *mode, bool bHardOpen/* = true*/)
	{
		Close();
		if (path == NULL || mode == NULL)
			return false;

		if (bHardOpen) {
			SAFE_FOPEN(fp, path, mode);
		}
		else {
			fp = fopen(path, mode);
		}

		strFileName = path;
		nLine = 0;
		nBuf = 1;
		bOver = false;
		return(fp != NULL);
	}
	bool File::Reopen(const char* model)
	{
		return Open(strFileName.c_str(), model);
	}

	char *File::GetLine(bool bPrecent/* = false*/)
	{
		if (!Good()) {
			lout_error("file open failed, can't GetLine!!");
		}

		if (bPrecent && nLine == 0)
			lout.Progress(fp, true, "Read File:");

		if (!pStrLine) {
			nBuf = 1;
			pStrLine = new char[MAX_SENTENCE_LEN];
		}

		bool bOk = GetLine(pStrLine, MAX_SENTENCE_LEN);
		if (bPrecent)
			lout.Progress(fp);

		while (bOver) {
			// re-alloc memory
			nBuf++;
			char *pBuf = new char[MAX_SENTENCE_LEN*nBuf];
			if (pBuf == NULL) {
				lout_error("[FILE GetLine] [Open Memory ERROR! (" << nBuf << "*" << MAX_SENTENCE_LEN << ")")
			}

			memcpy(pBuf, pStrLine, MAX_SENTENCE_LEN*(nBuf - 1)); //copy the buffer
			delete[]pStrLine;
			pStrLine = pBuf;
			bOk = GetLine(pStrLine + MAX_SENTENCE_LEN*(nBuf - 1) - 1, MAX_SENTENCE_LEN + 1);
			if (bPrecent)
				lout.Progress(fp);
		}


		if (bOk)
			return pStrLine;
		else
			return NULL;
	}

	bool File::GetLine(char *str, int maxLen/* = GS_SENTENCE_LEN*/)
	{
		bOver = false;
		char *p = fgets(str, maxLen, fp);
		if (p == NULL)
			return false;

		nLine++;
		int l = strlen(str);
		if (l >= maxLen - 1)
		{
			bOver = true;
			return false;
		}
		if (str[l - 1] == '\n')
			str[l - 1] = '\0';

#ifdef __linux
		// In linux, there may exist '\r\n' at the end of the line.
		// remove '\r'
		if (str[l-2] == '\r')
			str[l-2] = '\0';
#endif

		return true;
	}

	void File::Print(const char* p_pMessage, ...)
	{
		if (fp == NULL)
			return;

		char strBuffer[MAX_SENTENCE_LEN];
		va_list vaParams;
		va_start(vaParams, p_pMessage);
		//_vsnprintf(strBuffer,GS_SENTENCE_LEN,p_pMessage,vaParams);
		vfprintf(fp, p_pMessage, vaParams);
		va_end(vaParams);

		//fprintf(fp, "%s", strBuffer);

		fflush(fp);
	}
	/// scanf
	int File::Scanf(const char* p_pMessage, ...)
	{
		if (fp == NULL)
			return 0;

		va_list vaParams;

		va_start(vaParams, p_pMessage);
		int retval = vfscanf(fp, p_pMessage, vaParams);
		va_end(vaParams);


		return retval;
	}

	/// ��txt��write the head
	void ObjFile::WriteHeadT()
	{
		Print("nTotalNum = %d\n\n", m_nTotalNum);
		m_nCurNum = 0;
	}
	/// ��txt��read the head
	void ObjFile::ReadHeadT()
	{
		fscanf(File::fp, "nTotalNum = %d\n\n", &m_nTotalNum);
		m_nCurNum = 0;
	}

	/// ��txt��write the object
	void ObjFile::WriteObjT()
	{
		Print("nObj: %d\n", m_nCurNum);
		m_pObj->WriteT(*this);
		Print("\n");
		m_nCurNum++;
	}
	/// ��txt��read the object
	bool ObjFile::ReadObjT()
	{
		if (feof(fp))
			return false;
		if (m_nCurNum >= m_nTotalNum)
			return false;

		fscanf(File::fp, "nObj: %d\n", &m_nCurNum);

		m_pObj->ReadT(*this);
		fscanf(File::fp, "\n");
		return true;
	}

	/// ��bin��write the head
	void ObjFile::WriteHeadB()
	{
		fwrite(&m_nTotalNum, sizeof(m_nTotalNum), 1, fp);
		m_nCurNum = 0;
	}
	/// ��bin��read the head
	void ObjFile::ReadHeadB()
	{
		fread(&m_nTotalNum, sizeof(m_nTotalNum), 1, fp);
		m_nCurNum = 0;
	}
	/// ��bin��write the object
	void ObjFile::WriteObjB()
	{
		fwrite(&m_nCurNum, sizeof(m_nCurNum), 1, fp);
		m_pObj->WriteB(*this);
		m_nCurNum++;
	}
	/// ��bin��read the object
	bool ObjFile::ReadObjB()
	{
		if (feof(fp))
			return false;
		if (m_nCurNum >= m_nTotalNum)
			return false;

		fread(&m_nCurNum, sizeof(m_nCurNum), 1, fp);
		m_pObj->ReadB(*this);
		return true;
	}
}
