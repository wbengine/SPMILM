--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local ptb = require('data')
local stringx = require('pl.stringx')

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.

local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}
		
local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test, state_nbest
local vocab
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)

  return transfer_data(module)
end

local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  print("Calculate ppl on test data...")
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

local function run_nbest(fname, vocab_endid)
  local beginning_time = torch.tic()
  print(string.format("Rescore nbest data [vocab_endid=%d] [write to %s] ...", vocab_endid, fname))
  local f = io.open(fname, 'wt')
  reset_state(state_nbest)
  g_disable_dropout(model.rnns)
  local perp = 0
  local total_perp = 0
  local len = state_nbest.data:size(1)
  local sent = 0
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_nbest.data[i]
    local y = state_nbest.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
    
    if y[1] == vocab_endid then
      f:write(string.format('sent=%d   %f\n', sent+1, perp))
      total_perp = total_perp + perp
      perp = 0
      sent = sent + 1
    end
    
  end
  print("Test set perplexity : " .. g_f3(torch.exp(total_perp / (len - 1))))
  print("Rescore End, time = " .. g_d(torch.toc(beginning_time) / 60) .. ' minis')
  g_enable_dropout(model.rnns)
  f:close()
end

local function arg_isexist(s)
  for i = 1, #arg do
    if arg[i] == s then
      return true
    end
  end
  return false
end

local function arg_value(s, defv)
  local v = defv
  for i = 1, #arg-1 do
    if arg[i] == s then
      v = arg[i+1]
      break
    end
  end
  print(s .. '=' .. (v==nil and 'nil' or v))
  return v
end


local function write_param(path)
    print('write model to ', path)
    torch.save(path, model)
end

local function read_param(path)
    print('read mode from ', path)
    model = torch.load(path)
end

local function read_vocab(path)
  if path == nil then
    print('error [read_vocab] an nil path')
  end
  vocab = {}
  n = 0
  for line in io.lines(path) do
    local a = stringx.split(line)
    vocab[a[1]] = tonumber(a[2])
    n = n + 1
  end
  params.vocab_size = n
  print('vocab_size = ' .. n)
end


local function main()
  
  params.max_max_epoch = tonumber(arg_value('-epoch', '10'))
  params.rnn_size = tonumber(arg_value('-hidden', '200'))
  local corpus_vocab = arg_value('-vocab', nil)
  local corpus_train = arg_value('-train', nil)
  local corpus_valid = arg_value('-valid', nil)
  local corpus_test = arg_value('-test', nil)
  local lm_write = arg_value('-write', nil)
  local lm_read = arg_value('-read', nil)
  local lm_nbest = arg_value('-nbest', nil)
  local lm_score = arg_value('-score', nil)
  local gpu_num = {tonumber(arg_value('-gpu', '1'))}
  
  g_init_gpu(gpu_num)
  
  read_vocab(corpus_vocab)
  
  if corpus_train ~= nil then
    state_train = {data=transfer_data(ptb.traindataset(corpus_train, params.batch_size))}
    state_valid =  {data=transfer_data(ptb.validdataset(corpus_valid, params.batch_size))}
  end
  if corpus_test ~= nil then
    state_test =  {data=transfer_data(ptb.testdataset(corpus_test, params.batch_size))}
  end
  if lm_nbest ~= nil then
    state_nbest = {data=transfer_data(ptb.nbestdataset(lm_nbest, params.batch_size))}
  end
  
 
  -- setup lstm
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  
  -- read the exist lstm
  if lm_read ~= nil then  read_param(lm_read)  end
  
    
  --- begin training ...
  if corpus_train ~= nil then
    local step = 0
    local epoch = 0
    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    local words_per_step = params.seq_length * params.batch_size
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
    local perps
    while epoch < params.max_max_epoch do
      local perp = fp(state_train)
      if perps == nil then
	perps = torch.zeros(epoch_size):add(perp)
      end
      perps[step % epoch_size + 1] = perp
      step = step + 1
      bp(state_train)
      total_cases = total_cases + params.seq_length * params.batch_size
      epoch = step / epoch_size
      if step % torch.round(epoch_size / 10) == 10 then
	local wps = torch.floor(total_cases / torch.toc(start_time))
	local since_beginning = g_d(torch.toc(beginning_time) / 60)
	print('epoch = ' .. g_f3(epoch) ..
	      ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
	      ', wps = ' .. wps ..
	      ', dw:norm() = ' .. g_f3(model.norm_dw) ..
	      ', lr = ' ..  g_f3(params.lr) ..
	      ', since beginning = ' .. since_beginning .. ' mins.')
      end
      if step % epoch_size == 0 then
	run_valid()
	if epoch > params.max_epoch then
	    params.lr = params.lr / params.decay
	end
      end
      if step % 33 == 0 then
	cutorch.synchronize()
	collectgarbage()
      end
    end
    print("Training is over.")  
  end
  
  -- test ppl
  if corpus_test ~= nil then run_test() end
  
  -- get the nbest lis
  if lm_score ~= nil then run_nbest(lm_score, vocab['<eos>'])  end
  
  -- write model
  if lm_write ~= nil then  write_param(lm_write) end


end

main()
