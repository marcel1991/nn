local CriterionW = torch.class('nn.CriterionW')

function CriterionW:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function CriterionW:updateOutput(input, target, spWeights)
end

function CriterionW:forward(input, target, spWeights)
   return self:updateOutput(input, target, spWeights)
end

function CriterionW:backward(input, target, spWeights)
   return self:updateGradInput(input, target, spWeights)
end

function CriterionW:updateGradInput(input, target, spWeights)
end

function CriterionW:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function CriterionW:type(type, tensorCache)
   assert(type, 'Criterion: must provide a type to convert to')
   -- find all tensors and convert them
   for key,param in pairs(self) do
      self[key] = nn.utils.recursiveType(param, type, tensorCache)
   end
   return self
end

function CriterionW:float()
   return self:type('torch.FloatTensor')
end

function CriterionW:double()
   return self:type('torch.DoubleTensor')
end

function CriterionW:cuda()
   return self:type('torch.CudaTensor')
end

function Criterion:cudaHalf()
   return self:type('torch.CudaHalfTensor')
end

function CriterionW:__call__(input, target, spWeights)
   self.output = self:forward(input, target, spWeights)
   self.gradInput = self:backward(input, target, spWeights)
   return self.output, self.gradInput
end
