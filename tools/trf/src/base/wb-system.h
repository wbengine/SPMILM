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
* \date 2016-04-28
* \brief include all the wb-written modules
*/

#include "wb-log.h"
#include "wb-option.h"
#include "wb-file.h"
#include "wb-string.h"
#include "wb-vector.h"
#include "wb-lhash.h"
#include "wb-trie.h"
#include "wb-heap.h"
#include "wb-mat.h"
#include "wb-iter.h"
//#include "wb-win.h"
using namespace wb;

#include <iostream>
#include <iomanip> // such as std::setw(10)
#include <fstream>
#include <cmath>
#include <algorithm>
using namespace std;

/// define the main function
#define _wbMain int main(int _argc, char** _argv)

/// Error
#define ERROR(b) { cout<<b<<endl; return 0; }

