ĺP
ëť
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.14.02v1.14.0-rc1-22-gaf24dc91b588

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
dtype0	*
_output_shapes
: *
shape: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
h
inputPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
shape:˙˙˙˙˙˙˙˙˙

˝
=linear/final_linear/kernel/Initializer/truncated_normal/shapeConst*
valueB"
      *-
_class#
!loc:@linear/final_linear/kernel*
dtype0*
_output_shapes
:
°
<linear/final_linear/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *-
_class#
!loc:@linear/final_linear/kernel*
dtype0*
_output_shapes
: 
˛
>linear/final_linear/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×#<*-
_class#
!loc:@linear/final_linear/kernel*
dtype0*
_output_shapes
: 
ţ
Glinear/final_linear/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal=linear/final_linear/kernel/Initializer/truncated_normal/shape*
T0*-
_class#
!loc:@linear/final_linear/kernel*
dtype0*
_output_shapes

:

Ł
;linear/final_linear/kernel/Initializer/truncated_normal/mulMulGlinear/final_linear/kernel/Initializer/truncated_normal/TruncatedNormal>linear/final_linear/kernel/Initializer/truncated_normal/stddev*
T0*-
_class#
!loc:@linear/final_linear/kernel*
_output_shapes

:


7linear/final_linear/kernel/Initializer/truncated_normalAdd;linear/final_linear/kernel/Initializer/truncated_normal/mul<linear/final_linear/kernel/Initializer/truncated_normal/mean*
_output_shapes

:
*
T0*-
_class#
!loc:@linear/final_linear/kernel

linear/final_linear/kernel
VariableV2*-
_class#
!loc:@linear/final_linear/kernel*
dtype0*
_output_shapes

:
*
shape
:

Ř
!linear/final_linear/kernel/AssignAssignlinear/final_linear/kernel7linear/final_linear/kernel/Initializer/truncated_normal*
T0*-
_class#
!loc:@linear/final_linear/kernel*
_output_shapes

:


linear/final_linear/kernel/readIdentitylinear/final_linear/kernel*
_output_shapes

:
*
T0*-
_class#
!loc:@linear/final_linear/kernel
¤
*linear/final_linear/bias/Initializer/zerosConst*
valueB*    *+
_class!
loc:@linear/final_linear/bias*
dtype0*
_output_shapes
:

linear/final_linear/bias
VariableV2*
shape:*+
_class!
loc:@linear/final_linear/bias*
dtype0*
_output_shapes
:
Á
linear/final_linear/bias/AssignAssignlinear/final_linear/bias*linear/final_linear/bias/Initializer/zeros*
_output_shapes
:*
T0*+
_class!
loc:@linear/final_linear/bias

linear/final_linear/bias/readIdentitylinear/final_linear/bias*
_output_shapes
:*
T0*+
_class!
loc:@linear/final_linear/bias
~
linear/final_linear/MatMulMatMulinputlinear/final_linear/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

linear/final_linear/BiasAddBiasAddlinear/final_linear/MatMullinear/final_linear/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
P
ShapeShapelinear/final_linear/BiasAdd*
T0*
_output_shapes
:
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
­
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
K
CastCaststrided_slice*

SrcT0*
_output_shapes
: *

DstT0

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_aa59ffa703f341f494f6eafdc654b543/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ą
save/SaveV2/tensor_namesConst"/device:CPU:0*V
valueMBKBglobal_stepBlinear/final_linear/biasBlinear/final_linear/kernel*
dtype0*
_output_shapes
:
x
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
Č
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_steplinear/final_linear/biaslinear/final_linear/kernel"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
´
save/RestoreV2/tensor_namesConst"/device:CPU:0*V
valueMBKBglobal_stepBlinear/final_linear/biasBlinear/final_linear/kernel*
dtype0*
_output_shapes
:
{
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B *
dtype0*
_output_shapes
:
Š
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
_output_shapes
: 

save/Assign_1Assignlinear/final_linear/biassave/RestoreV2:1*
T0*+
_class!
loc:@linear/final_linear/bias*
_output_shapes
:

save/Assign_2Assignlinear/final_linear/kernelsave/RestoreV2:2*
_output_shapes

:
*
T0*-
_class#
!loc:@linear/final_linear/kernel
H
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"˘
	variables
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
Ą
linear/final_linear/kernel:0!linear/final_linear/kernel/Assign!linear/final_linear/kernel/read:029linear/final_linear/kernel/Initializer/truncated_normal:08

linear/final_linear/bias:0linear/final_linear/bias/Assignlinear/final_linear/bias/read:02,linear/final_linear/bias/Initializer/zeros:08"%
saved_model_main_op


group_deps"Đ
trainable_variables¸ľ
Ą
linear/final_linear/kernel:0!linear/final_linear/kernel/Assign!linear/final_linear/kernel/read:029linear/final_linear/kernel/Initializer/truncated_normal:08

linear/final_linear/bias:0linear/final_linear/bias/Assignlinear/final_linear/bias/read:02,linear/final_linear/bias/Initializer/zeros:08*
serving_default
'
input
input:0˙˙˙˙˙˙˙˙˙
>
output4
linear/final_linear/BiasAdd:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict