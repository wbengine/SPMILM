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




/**
* \file
* \author WangBin
* \date 2016-05-05
* \brief define the file class
*/


#pragma once
#include "iostream"
#include "fstream"
using namespace std;

#include <stdio.h>
#include <time.h>
#include <cstdarg>	//include va_list

#include "wb-win.h"
#include "wb-log.h"

/// file open
#define SAFE_FOPEN(pfile, path, mode) {\
	if ( !path || !(pfile = fopen(path, mode)) ) {cout<<"File Open Failed: "<<path<<endl; exit(0);}} 
/// file open
#define SAFE_FMOPEN(file, path, mode) {\
	file.open(path, mode); if (!file.good()) {cout<<"File Open Failed: "<<path<<endl; exit(0);}} 
/// file open
#define SAFE_FSOPEN(file, path) {\
	file.open(path); if (!file.good()) {cout<<"File Open Failed: "<<path<<endl; exit(0);}} 
/// file close
#define SAFE_FCLOSE(fp) { if(fp) {fclose(fp); fp=NULL;} } 

/// the maximum length of string read from file once
#define MAX_SENTENCE_LEN	32*1024 

/// Write a value to file
#define WRITE_VALUE(x,fp) fwrite(&(x),sizeof(x),1,fp)
/// Read a value from file
#define READ_VALUE(x,fp) fread(&(x),sizeof(x),1,fp)
/// fprintf a value to file
#define FPRINT_VALUE(x, fp) fprintf(fp, #x"=%d\n", (x))
/// fscanf a value from file
#define FSCANF_VALUE(x, fp) fscanf(fp, #x"=%d\n", &(x))

namespace wb
{
	/*!
	 * \author WangBin
	 * \date 2016-05-05
	 * \brief file class.
	 * \details
	 * There are several convenient functions in this class:
	 *	- function GetLine. It can read a line in a text file. For example:
	 *		\code{.cpp}
	 *		char *pline = NULL;
	 *		while (pline = file.GetLine()) {
	 *		 ... process the line ...
	 *		}
	 *		\endcode
	 *	- function PrintArray. It can print a array to into a line of a text file. For example:
	 *		\code{.cpp}
	 *		float a[10] = {1,2,3};
	 *		file.PrintArray("%f", a, 10);
	 *		\endcode
	 */
	class File
	{
	public:
		FILE *fp;  ///< file pointer
		int nLine; ///< the number of reading from file

		char *pStrLine; ///< store the string get from file
		string strFileName; ///< stroe the file name

		bool bOver; ///< record the if the buffer is overflow
		short nBuf; ///< record the buffer is nBuf times of GS_SENTENCE_LEN

	public:
		/// constructor
		File() :fp(NULL), pStrLine(NULL) { fp = NULL; bOver = false; nBuf = 1; }
		/// constructor
		File(const char *path, const char *mode, bool bHardOpen = true) :fp(NULL), pStrLine(NULL) { Open(path, mode, bHardOpen); }
		/// destructor
		~File() { SAFE_FCLOSE(fp); delete[]pStrLine; }
		/// Open file
		/**
		 * \param [in] path the file path
		 * \param [in] mode open mode {'w', 'r', 't', 'b', 'a'}
		 * \param [in] bHardOpen if ture, report error if opening the file fails.
		 * \return if open the file successfully.
		 */
		virtual bool Open(const char *path, const char *mode, bool bHardOpen = true);
		/// re-open the file
		virtual bool Reopen(const char* model);
		/// close the file
		virtual void Close() { SAFE_FCLOSE(fp); }
		/// transform the class to FILE pointer
		operator FILE* () { return fp; }
		/// Read a line into the buffer
		/**
		* \param [in] bPrecent if true, then print the precent in the tille
		* \return return the string, which can be accepted by char* variables.
		*/
		virtual char *GetLine(bool bPrecent = false);
		/// read a line into buffer str.
		virtual bool GetLine(char *str, int maxLen = MAX_SENTENCE_LEN);
		/// print
		virtual void Print(const char* p_pMessage, ...);
		/// scanf
		virtual int Scanf(const char* p_pMessage, ...);
		/// clean buffer
		void Flush()  { fflush(fp); }
		/// reset position
		/** using _fseeki64 to reset position */
		void Reset() { _fseeki64(fp, 0, SEEK_SET); nLine = 0; }
		/// return if the file is accessible.
		bool Good() const { return !(fp == NULL); }
		/// print a array into file
		template <typename TYPE>
		void PrintArray(const char* pformat, TYPE* pbuf, int num)
		{
			for (int i = 0; i<num; i++) {
				Print(pformat, *pbuf);
				putc(' ', fp);
				pbuf++;
			}
			putc('\n', fp);
		}
	};

	/// base class used to derive. It providing the vritual function of write and read
	class IO_Obj
	{
	public:
		/// write to the txt file
		virtual void WriteT(File &file) {};
		/// read from the txt file
		virtual void ReadT(File &file) {};
		/// write to binary file
		virtual void WriteB(File &file) {};
		/// read from binary file
		virtual void ReadB(File &file) {};
	};

	/// used to read more than one objects
	class ObjFile : public File
	{
	public:
		int m_nTotalNum; ///< object number
		int m_nCurNum;   ///< current object number
		IO_Obj *m_pObj; ///< object pointer
	public:
		/// constructor
		ObjFile() :File(), m_nTotalNum(0), m_nCurNum(-1){}
		/// constructor
		ObjFile(const char *path, const char *mode) :File(path, mode), m_nTotalNum(0), m_nCurNum(-1){}
		/// ��txt��write the head
		void WriteHeadT();
		/// ��txt��read the head
		void ReadHeadT();
		/// ��txt��write the object
		void WriteObjT();
		/// ��txt��read the object
		bool ReadObjT();
		/// ��bin��write the head
		void WriteHeadB();
		/// ��bin��read the head
		void ReadHeadB();
		/// ��bin��write the object
		void WriteObjB();
		/// ��bin��read the object
		bool ReadObjB();
	};

}




