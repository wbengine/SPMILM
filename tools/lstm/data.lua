--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = "./data/"

--vocab_idx = 0
--local vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

local function load_data(fname)
  local words = 0
  local sents = 0
  
  if fname == nil then
    print('error [load_data] a nill input')
  end
  for line in io.lines(fname) do
    local data = stringx.split(line)
    words = words + #data
    sents = sents + 1
  end
  print(string.format("Loading %s, sents = %d words = %d", fname, sents, words))
  
  local x = torch.zeros(words)
  local n = 0
  for line in io.lines(fname) do
    local data = stringx.split(line)
    for i = 1, #data do
      n = n + 1
      x[n] = tonumber(data[i])
    end
  end
  
  print('Load End n=' .. n)
  return x
end

local function traindataset(fname, batch_size)
   local x = load_data(fname)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(fname, batch_size)
   local x = load_data(fname)
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   return x
end

local function validdataset(fname, batch_size)
   local x = load_data(fname)
   x = replicate(x, batch_size)
   return x
end

local function nbestdataset(fname, batch_size)
   local x = load_data(fname)
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   return x
end

return {traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
	nbestdataset=nbestdataset}
