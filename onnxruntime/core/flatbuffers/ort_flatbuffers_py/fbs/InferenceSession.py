# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class InferenceSession(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = InferenceSession()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsInferenceSession(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def InferenceSessionBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # InferenceSession
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # InferenceSession
    def OrtVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # InferenceSession
    def Model(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.Model import Model
            obj = Model()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # InferenceSession
    def KernelTypeStrResolver(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.KernelTypeStrResolver import KernelTypeStrResolver
            obj = KernelTypeStrResolver()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def InferenceSessionStart(builder):
    builder.StartObject(4)

def Start(builder):
    InferenceSessionStart(builder)

def InferenceSessionAddOrtVersion(builder, ortVersion):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(ortVersion), 0)

def AddOrtVersion(builder, ortVersion):
    InferenceSessionAddOrtVersion(builder, ortVersion)

def InferenceSessionAddModel(builder, model):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(model), 0)

def AddModel(builder, model):
    InferenceSessionAddModel(builder, model)

def InferenceSessionAddKernelTypeStrResolver(builder, kernelTypeStrResolver):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(kernelTypeStrResolver), 0)

def AddKernelTypeStrResolver(builder, kernelTypeStrResolver):
    InferenceSessionAddKernelTypeStrResolver(builder, kernelTypeStrResolver)

def InferenceSessionEnd(builder):
    return builder.EndObject()

def End(builder):
    return InferenceSessionEnd(builder)
