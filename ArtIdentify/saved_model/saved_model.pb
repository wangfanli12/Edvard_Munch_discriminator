??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
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
delete_old_dirsbool(?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
conv2d_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_nameconv2d_17/kernel
~
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*'
_output_shapes
:?*
dtype0
u
conv2d_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_17/bias
n
"conv2d_17/bias/Read/ReadVariableOpReadVariableOpconv2d_17/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_16/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_16/gamma
?
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_16/beta
?
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_16/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_16/moving_mean
?
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_16/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_16/moving_variance
?
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_18/kernel

$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_18/bias
n
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_17/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_17/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_17/beta
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_17/moving_mean
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_17/moving_variance
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_19/kernel

$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_18/gamma
?
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_18/beta
?
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:?*
dtype0
?
"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_18/moving_mean
?
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:?*
dtype0
?
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_18/moving_variance
?
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:?*
dtype0
?
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_20/kernel
~
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_19/gamma
?
0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_19/beta
?
/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes
:@*
dtype0
?
"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_19/moving_mean
?
6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes
:@*
dtype0
?
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_19/moving_variance
?
:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes
:@*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
??*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2d_17/kernel/m
?
+Adam/conv2d_17/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/m*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_17/bias/m
|
)Adam/conv2d_17/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_16/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_16/gamma/m
?
7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_16/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_16/beta/m
?
6Adam/batch_normalization_16/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_18/kernel/m
?
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_18/bias/m
|
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_17/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_17/gamma/m
?
7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_17/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_17/beta/m
?
6Adam/batch_normalization_17/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_19/kernel/m
?
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_19/bias/m
|
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_18/gamma/m
?
7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_18/beta/m
?
6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_20/kernel/m
?
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_19/gamma/m
?
7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_19/beta/m
?
6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_4/kernel/m
?
)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
??*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/conv2d_17/kernel/v
?
+Adam/conv2d_17/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/kernel/v*'
_output_shapes
:?*
dtype0
?
Adam/conv2d_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_17/bias/v
|
)Adam/conv2d_17/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_17/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_16/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_16/gamma/v
?
7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_16/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_16/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_16/beta/v
?
6Adam/batch_normalization_16/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_16/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_18/kernel/v
?
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_18/bias/v
|
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_17/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_17/gamma/v
?
7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_17/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_17/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_17/beta/v
?
6Adam/batch_normalization_17/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_17/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_19/kernel/v
?
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_19/bias/v
|
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_18/gamma/v
?
7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes	
:?*
dtype0
?
"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_18/beta/v
?
6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_20/kernel/v
?
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_19/gamma/v
?
7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes
:@*
dtype0
?
"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_19/beta/v
?
6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes
:@*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/dense_4/kernel/v
?
)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
??*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?v
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?v
value?vB?u B?u
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
R
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
h

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
R
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
R
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?!m?"m?)m?*m?4m?5m?<m?=m?Gm?Hm?Om?Pm?Zm?[m?jm?km?v?v?!v?"v?)v?*v?4v?5v?<v?=v?Gv?Hv?Ov?Pv?Zv?[v?jv?kv?
?
0
1
!2
"3
#4
$5
)6
*7
48
59
610
711
<12
=13
G14
H15
I16
J17
O18
P19
Z20
[21
\22
]23
j24
k25
?
0
1
!2
"3
)4
*5
46
57
<8
=9
G10
H11
O12
P13
Z14
[15
j16
k17
 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_17/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_17/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_16/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_16/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_16/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_16/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
#2
$3

!0
"1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_17/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_17/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_17/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_17/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51
62
73

40
51
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
I2
J3

G0
H1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

O0
P1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
\2
]3

Z0
[1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
8
#0
$1
62
73
I4
J5
\6
]7
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 

#0
$1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 
 
 
 

I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

\0
]1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_17/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_16/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_17/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_19/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_17/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_17/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_16/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_16/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_17/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_17/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_19/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_17_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_17_inputconv2d_17/kernelconv2d_17/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_19/kernelconv2d_19/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_4/kerneldense_4/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_18159
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_17/kernel/Read/ReadVariableOp"conv2d_17/bias/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_17/kernel/m/Read/ReadVariableOp)Adam/conv2d_17/bias/m/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_16/beta/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_17/beta/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp+Adam/conv2d_17/kernel/v/Read/ReadVariableOp)Adam/conv2d_17/bias/v/Read/ReadVariableOp7Adam/batch_normalization_16/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_16/beta/v/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp7Adam/batch_normalization_17/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_17/beta/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_19382
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_17/kernelconv2d_17/biasbatch_normalization_16/gammabatch_normalization_16/beta"batch_normalization_16/moving_mean&batch_normalization_16/moving_varianceconv2d_18/kernelconv2d_18/biasbatch_normalization_17/gammabatch_normalization_17/beta"batch_normalization_17/moving_mean&batch_normalization_17/moving_varianceconv2d_19/kernelconv2d_19/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_variancedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_17/kernel/mAdam/conv2d_17/bias/m#Adam/batch_normalization_16/gamma/m"Adam/batch_normalization_16/beta/mAdam/conv2d_18/kernel/mAdam/conv2d_18/bias/m#Adam/batch_normalization_17/gamma/m"Adam/batch_normalization_17/beta/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/conv2d_17/kernel/vAdam/conv2d_17/bias/v#Adam/batch_normalization_16/gamma/v"Adam/batch_normalization_16/beta/vAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/v#Adam/batch_normalization_17/gamma/v"Adam/batch_normalization_17/beta/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/vAdam/dense_4/kernel/vAdam/dense_4/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_19605??
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17233

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_17_layer_call_fn_18500

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18593

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18746

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_19003

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17333w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17030

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@@?*
alpha%
?#<h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18495

inputs9
conv2d_readvariableop_resource:?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_16880
conv2d_17_inputP
5sequential_5_conv2d_17_conv2d_readvariableop_resource:?E
6sequential_5_conv2d_17_biasadd_readvariableop_resource:	?J
;sequential_5_batch_normalization_16_readvariableop_resource:	?L
=sequential_5_batch_normalization_16_readvariableop_1_resource:	?[
Lsequential_5_batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_5_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?Q
5sequential_5_conv2d_18_conv2d_readvariableop_resource:??E
6sequential_5_conv2d_18_biasadd_readvariableop_resource:	?J
;sequential_5_batch_normalization_17_readvariableop_resource:	?L
=sequential_5_batch_normalization_17_readvariableop_1_resource:	?[
Lsequential_5_batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_5_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?Q
5sequential_5_conv2d_19_conv2d_readvariableop_resource:??E
6sequential_5_conv2d_19_biasadd_readvariableop_resource:	?J
;sequential_5_batch_normalization_18_readvariableop_resource:	?L
=sequential_5_batch_normalization_18_readvariableop_1_resource:	?[
Lsequential_5_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	?]
Nsequential_5_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	?P
5sequential_5_conv2d_20_conv2d_readvariableop_resource:?@D
6sequential_5_conv2d_20_biasadd_readvariableop_resource:@I
;sequential_5_batch_normalization_19_readvariableop_resource:@K
=sequential_5_batch_normalization_19_readvariableop_1_resource:@Z
Lsequential_5_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_5_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:@G
3sequential_5_dense_4_matmul_readvariableop_resource:
??B
4sequential_5_dense_4_biasadd_readvariableop_resource:
identity??Csequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_16/ReadVariableOp?4sequential_5/batch_normalization_16/ReadVariableOp_1?Csequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_17/ReadVariableOp?4sequential_5/batch_normalization_17/ReadVariableOp_1?Csequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_18/ReadVariableOp?4sequential_5/batch_normalization_18/ReadVariableOp_1?Csequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp?Esequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?2sequential_5/batch_normalization_19/ReadVariableOp?4sequential_5/batch_normalization_19/ReadVariableOp_1?-sequential_5/conv2d_17/BiasAdd/ReadVariableOp?,sequential_5/conv2d_17/Conv2D/ReadVariableOp?-sequential_5/conv2d_18/BiasAdd/ReadVariableOp?,sequential_5/conv2d_18/Conv2D/ReadVariableOp?-sequential_5/conv2d_19/BiasAdd/ReadVariableOp?,sequential_5/conv2d_19/Conv2D/ReadVariableOp?-sequential_5/conv2d_20/BiasAdd/ReadVariableOp?,sequential_5/conv2d_20/Conv2D/ReadVariableOp?+sequential_5/dense_4/BiasAdd/ReadVariableOp?*sequential_5/dense_4/MatMul/ReadVariableOp?
,sequential_5/conv2d_17/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
sequential_5/conv2d_17/Conv2DConv2Dconv2d_17_input4sequential_5/conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
-sequential_5/conv2d_17/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_5/conv2d_17/BiasAddBiasAdd&sequential_5/conv2d_17/Conv2D:output:05sequential_5/conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
%sequential_5/leaky_re_lu_17/LeakyRelu	LeakyRelu'sequential_5/conv2d_17/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
2sequential_5/batch_normalization_16/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_16/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Csequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Esequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_16/FusedBatchNormV3FusedBatchNormV33sequential_5/leaky_re_lu_17/LeakyRelu:activations:0:sequential_5/batch_normalization_16/ReadVariableOp:value:0<sequential_5/batch_normalization_16/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
,sequential_5/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_5/conv2d_18/Conv2DConv2D8sequential_5/batch_normalization_16/FusedBatchNormV3:y:04sequential_5/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
-sequential_5/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_5/conv2d_18/BiasAddBiasAdd&sequential_5/conv2d_18/Conv2D:output:05sequential_5/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
%sequential_5/leaky_re_lu_18/LeakyRelu	LeakyRelu'sequential_5/conv2d_18/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
2sequential_5/batch_normalization_17/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_17/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Csequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Esequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_17/FusedBatchNormV3FusedBatchNormV33sequential_5/leaky_re_lu_18/LeakyRelu:activations:0:sequential_5/batch_normalization_17/ReadVariableOp:value:0<sequential_5/batch_normalization_17/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
,sequential_5/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_5/conv2d_19/Conv2DConv2D8sequential_5/batch_normalization_17/FusedBatchNormV3:y:04sequential_5/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
-sequential_5/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_5/conv2d_19/BiasAddBiasAdd&sequential_5/conv2d_19/Conv2D:output:05sequential_5/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@??
%sequential_5/leaky_re_lu_19/LeakyRelu	LeakyRelu'sequential_5/conv2d_19/BiasAdd:output:0*0
_output_shapes
:?????????@@?*
alpha%
?#<?
2sequential_5/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_18_readvariableop_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Csequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Esequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
4sequential_5/batch_normalization_18/FusedBatchNormV3FusedBatchNormV33sequential_5/leaky_re_lu_19/LeakyRelu:activations:0:sequential_5/batch_normalization_18/ReadVariableOp:value:0<sequential_5/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( ?
,sequential_5/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_5_conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
sequential_5/conv2d_20/Conv2DConv2D8sequential_5/batch_normalization_18/FusedBatchNormV3:y:04sequential_5/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
-sequential_5/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_5_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_5/conv2d_20/BiasAddBiasAdd&sequential_5/conv2d_20/Conv2D:output:05sequential_5/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
%sequential_5/leaky_re_lu_20/LeakyRelu	LeakyRelu'sequential_5/conv2d_20/BiasAdd:output:0*/
_output_shapes
:?????????  @*
alpha%
?#<?
2sequential_5/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_5_batch_normalization_19_readvariableop_resource*
_output_shapes
:@*
dtype0?
4sequential_5/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_5_batch_normalization_19_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Csequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_5_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Esequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_5_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
4sequential_5/batch_normalization_19/FusedBatchNormV3FusedBatchNormV33sequential_5/leaky_re_lu_20/LeakyRelu:activations:0:sequential_5/batch_normalization_19/ReadVariableOp:value:0<sequential_5/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( m
sequential_5/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
sequential_5/flatten_4/ReshapeReshape8sequential_5/batch_normalization_19/FusedBatchNormV3:y:0%sequential_5/flatten_4/Const:output:0*
T0*)
_output_shapes
:????????????
sequential_5/dropout_4/IdentityIdentity'sequential_5/flatten_4/Reshape:output:0*
T0*)
_output_shapes
:????????????
*sequential_5/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_5/dense_4/MatMulMatMul(sequential_5/dropout_4/Identity:output:02sequential_5/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_5/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_5/dense_4/BiasAddBiasAdd%sequential_5/dense_4/MatMul:product:03sequential_5/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_5/dense_4/SoftmaxSoftmax%sequential_5/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????u
IdentityIdentity&sequential_5/dense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpD^sequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_16/ReadVariableOp5^sequential_5/batch_normalization_16/ReadVariableOp_1D^sequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_17/ReadVariableOp5^sequential_5/batch_normalization_17/ReadVariableOp_1D^sequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_18/ReadVariableOp5^sequential_5/batch_normalization_18/ReadVariableOp_1D^sequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_5/batch_normalization_19/ReadVariableOp5^sequential_5/batch_normalization_19/ReadVariableOp_1.^sequential_5/conv2d_17/BiasAdd/ReadVariableOp-^sequential_5/conv2d_17/Conv2D/ReadVariableOp.^sequential_5/conv2d_18/BiasAdd/ReadVariableOp-^sequential_5/conv2d_18/Conv2D/ReadVariableOp.^sequential_5/conv2d_19/BiasAdd/ReadVariableOp-^sequential_5/conv2d_19/Conv2D/ReadVariableOp.^sequential_5/conv2d_20/BiasAdd/ReadVariableOp-^sequential_5/conv2d_20/Conv2D/ReadVariableOp,^sequential_5/dense_4/BiasAdd/ReadVariableOp+^sequential_5/dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Csequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_16/ReadVariableOp2sequential_5/batch_normalization_16/ReadVariableOp2l
4sequential_5/batch_normalization_16/ReadVariableOp_14sequential_5/batch_normalization_16/ReadVariableOp_12?
Csequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_17/ReadVariableOp2sequential_5/batch_normalization_17/ReadVariableOp2l
4sequential_5/batch_normalization_17/ReadVariableOp_14sequential_5/batch_normalization_17/ReadVariableOp_12?
Csequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_18/ReadVariableOp2sequential_5/batch_normalization_18/ReadVariableOp2l
4sequential_5/batch_normalization_18/ReadVariableOp_14sequential_5/batch_normalization_18/ReadVariableOp_12?
Csequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2?
Esequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_5/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_5/batch_normalization_19/ReadVariableOp2sequential_5/batch_normalization_19/ReadVariableOp2l
4sequential_5/batch_normalization_19/ReadVariableOp_14sequential_5/batch_normalization_19/ReadVariableOp_12^
-sequential_5/conv2d_17/BiasAdd/ReadVariableOp-sequential_5/conv2d_17/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_17/Conv2D/ReadVariableOp,sequential_5/conv2d_17/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_18/BiasAdd/ReadVariableOp-sequential_5/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_18/Conv2D/ReadVariableOp,sequential_5/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_19/BiasAdd/ReadVariableOp-sequential_5/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_19/Conv2D/ReadVariableOp,sequential_5/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_5/conv2d_20/BiasAdd/ReadVariableOp-sequential_5/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_5/conv2d_20/Conv2D/ReadVariableOp,sequential_5/conv2d_20/Conv2D/ReadVariableOp2Z
+sequential_5/dense_4/BiasAdd/ReadVariableOp+sequential_5/dense_4/BiasAdd/ReadVariableOp2X
*sequential_5/dense_4/MatMul/ReadVariableOp*sequential_5/dense_4/MatMul/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18611

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17623

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17683

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17183

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153

inputs9
conv2d_readvariableop_resource:?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_17952
conv2d_17_input"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
??

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_17840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?
e
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_18811

inputs
identity`
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@@?*
alpha%
?#<h
IdentityIdentityLeakyRelu:activations:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17125

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_19_layer_call_fn_19016

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17503w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16902

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18575

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_18_layer_call_fn_18850

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17283x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?

?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_16_layer_call_fn_18518

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16902?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_18159
conv2d_17_input"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
??

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_16880o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?
?
6__inference_batch_normalization_18_layer_call_fn_18863

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17563x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
)__inference_conv2d_18_layer_call_fn_18638

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18728

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_20_layer_call_fn_18959

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_19135

inputs
unknown:
??
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_17369o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_17_layer_call_fn_18485

inputs"
unknown:?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16997

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_19_layer_call_fn_18990

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17125?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
??
? 
__inference__traced_save_19382
file_prefix/
+savev2_conv2d_17_kernel_read_readvariableop-
)savev2_conv2d_17_bias_read_readvariableop;
7savev2_batch_normalization_16_gamma_read_readvariableop:
6savev2_batch_normalization_16_beta_read_readvariableopA
=savev2_batch_normalization_16_moving_mean_read_readvariableopE
Asavev2_batch_normalization_16_moving_variance_read_readvariableop/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop;
7savev2_batch_normalization_17_gamma_read_readvariableop:
6savev2_batch_normalization_17_beta_read_readvariableopA
=savev2_batch_normalization_17_moving_mean_read_readvariableopE
Asavev2_batch_normalization_17_moving_variance_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_17_kernel_m_read_readvariableop4
0savev2_adam_conv2d_17_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop6
2savev2_adam_conv2d_17_kernel_v_read_readvariableop4
0savev2_adam_conv2d_17_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_16_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_16_beta_v_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_17_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_17_beta_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?&
value?&B?&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_17_kernel_read_readvariableop)savev2_conv2d_17_bias_read_readvariableop7savev2_batch_normalization_16_gamma_read_readvariableop6savev2_batch_normalization_16_beta_read_readvariableop=savev2_batch_normalization_16_moving_mean_read_readvariableopAsavev2_batch_normalization_16_moving_variance_read_readvariableop+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop7savev2_batch_normalization_17_gamma_read_readvariableop6savev2_batch_normalization_17_beta_read_readvariableop=savev2_batch_normalization_17_moving_mean_read_readvariableopAsavev2_batch_normalization_17_moving_variance_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_17_kernel_m_read_readvariableop0savev2_adam_conv2d_17_bias_m_read_readvariableop>savev2_adam_batch_normalization_16_gamma_m_read_readvariableop=savev2_adam_batch_normalization_16_beta_m_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop>savev2_adam_batch_normalization_17_gamma_m_read_readvariableop=savev2_adam_batch_normalization_17_beta_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop2savev2_adam_conv2d_17_kernel_v_read_readvariableop0savev2_adam_conv2d_17_bias_v_read_readvariableop>savev2_adam_batch_normalization_16_gamma_v_read_readvariableop=savev2_adam_batch_normalization_16_beta_v_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop>savev2_adam_batch_normalization_17_gamma_v_read_readvariableop=savev2_adam_batch_normalization_17_beta_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?:?:?:?:??:?:?:?:?:?:??:?:?:?:?:?:?@:@:@:@:@:@:
??:: : : : : : : : : :?:?:?:?:??:?:?:?:??:?:?:?:?@:@:@:@:
??::?:?:?:?:??:?:?:?:??:?:?:?:?@:@:@:@:
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!	

_output_shapes	
:?:!


_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :-$)
'
_output_shapes
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:!'

_output_shapes	
:?:.(*
(
_output_shapes
:??:!)

_output_shapes	
:?:!*

_output_shapes	
:?:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:-0)
'
_output_shapes
:?@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:&4"
 
_output_shapes
:
??: 5

_output_shapes
::-6)
'
_output_shapes
:?:!7

_output_shapes	
:?:!8

_output_shapes	
:?:!9

_output_shapes	
:?:.:*
(
_output_shapes
:??:!;

_output_shapes	
:?:!<

_output_shapes	
:?:!=

_output_shapes	
:?:.>*
(
_output_shapes
:??:!?

_output_shapes	
:?:!@

_output_shapes	
:?:!A

_output_shapes	
:?:-B)
'
_output_shapes
:?@: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:&F"
 
_output_shapes
:
??: G

_output_shapes
::H

_output_shapes
: 
?y
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18371

inputsC
(conv2d_17_conv2d_readvariableop_resource:?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_16_readvariableop_resource:	??
0batch_normalization_16_readvariableop_1_resource:	?N
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_18_conv2d_readvariableop_resource:??8
)conv2d_18_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_19_conv2d_readvariableop_resource:??8
)conv2d_19_biasadd_readvariableop_resource:	?=
.batch_normalization_18_readvariableop_resource:	??
0batch_normalization_18_readvariableop_1_resource:	?N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_20_conv2d_readvariableop_resource:?@7
)conv2d_20_biasadd_readvariableop_resource:@<
.batch_normalization_19_readvariableop_resource:@>
0batch_normalization_19_readvariableop_1_resource:@M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:@:
&dense_4_matmul_readvariableop_resource:
??5
'dense_4_biasadd_readvariableop_resource:
identity??6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1?6batch_normalization_18/FusedBatchNormV3/ReadVariableOp?8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_18/ReadVariableOp?'batch_normalization_18/ReadVariableOp_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
leaky_re_lu_17/LeakyRelu	LeakyReluconv2d_17/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_17/LeakyRelu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_18/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
leaky_re_lu_18/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_18/LeakyRelu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_19/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@??
leaky_re_lu_19/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*0
_output_shapes
:?????????@@?*
alpha%
?#<?
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_19/LeakyRelu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( ?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv2d_20/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
leaky_re_lu_20/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:?????????  @*
alpha%
?#<?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_20/LeakyRelu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( `
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_4/ReshapeReshape+batch_normalization_19/FusedBatchNormV3:y:0flatten_4/Const:output:0*
T0*)
_output_shapes
:???????????n
dropout_4/IdentityIdentityflatten_4/Reshape:output:0*
T0*)
_output_shapes
:????????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_4/MatMulMatMuldropout_4/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp7^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17283

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_18557

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17683z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17333

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_17_layer_call_fn_18671

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16966?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17563

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18782

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18954

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18935

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19052

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19088

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_17369

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17094

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?J
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18023
conv2d_17_input*
conv2d_17_17955:?
conv2d_17_17957:	?+
batch_normalization_16_17961:	?+
batch_normalization_16_17963:	?+
batch_normalization_16_17965:	?+
batch_normalization_16_17967:	?+
conv2d_18_17970:??
conv2d_18_17972:	?+
batch_normalization_17_17976:	?+
batch_normalization_17_17978:	?+
batch_normalization_17_17980:	?+
batch_normalization_17_17982:	?+
conv2d_19_17985:??
conv2d_19_17987:	?+
batch_normalization_18_17991:	?+
batch_normalization_18_17993:	?+
batch_normalization_18_17995:	?+
batch_normalization_18_17997:	?*
conv2d_20_18000:?@
conv2d_20_18002:@*
batch_normalization_19_18006:@*
batch_normalization_19_18008:@*
batch_normalization_19_18010:@*
batch_normalization_19_18012:@!
dense_4_18017:
??
dense_4_18019:
identity??.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_17955conv2d_17_17957*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153?
leaky_re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0batch_normalization_16_17961batch_normalization_16_17963batch_normalization_16_17965batch_normalization_16_17967*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17183?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_18_17970conv2d_18_17972*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203?
leaky_re_lu_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0batch_normalization_17_17976batch_normalization_17_17978batch_normalization_17_17980batch_normalization_17_17982*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17233?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_19_17985conv2d_19_17987*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253?
leaky_re_lu_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0batch_normalization_18_17991batch_normalization_18_17993batch_normalization_18_17995batch_normalization_18_17997*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17283?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_20_18000conv2d_20_18002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303?
leaky_re_lu_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0batch_normalization_19_18006batch_normalization_19_18008batch_normalization_19_18010batch_normalization_19_18012*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17333?
flatten_4/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349?
dropout_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17356?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_18017dense_4_18019*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_17369w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?

?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_18648

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????j
IdentityIdentityBiasAdd:output:0^NoOp*
T0*2
_output_shapes 
:????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18899

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_18964

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_19_layer_call_fn_18977

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17094?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_19146

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_19126

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????q
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????k
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_19099

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_18216

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
??

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_17376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_17356

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????  @*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17503

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_17_layer_call_fn_18684

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16997?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19034

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_16_layer_call_fn_18531

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16933?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_18658

inputs
identityb
	LeakyRelu	LeakyReluinputs*2
_output_shapes 
:????????????*
alpha%
?#<j
IdentityIdentityLeakyRelu:activations:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_4_layer_call_fn_19104

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17356b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214

inputs
identityb
	LeakyRelu	LeakyReluinputs*2
_output_shapes 
:????????????*
alpha%
?#<j
IdentityIdentityLeakyRelu:activations:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_18_layer_call_fn_18837

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17061?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_16_layer_call_fn_18544

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17183z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18764

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
is_training( n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_18505

inputs
identityb
	LeakyRelu	LeakyReluinputs*2
_output_shapes 
:????????????*
alpha%
?#<j
IdentityIdentityLeakyRelu:activations:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_4_layer_call_fn_19093

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
)__inference_conv2d_20_layer_call_fn_18944

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_19114

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_19_layer_call_fn_18806

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_18273

inputs"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
??

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_17840o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_re_lu_18_layer_call_fn_18653

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214k
IdentityIdentityPartitionedCall:output:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_18_layer_call_fn_18824

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17030?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_18801

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
e
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164

inputs
identityb
	LeakyRelu	LeakyReluinputs*2
_output_shapes 
:????????????*
alpha%
?#<j
IdentityIdentityLeakyRelu:activations:0*
T0*2
_output_shapes 
:????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????????:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18881

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_19_layer_call_fn_18791

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
??
?/
!__inference__traced_restore_19605
file_prefix<
!assignvariableop_conv2d_17_kernel:?0
!assignvariableop_1_conv2d_17_bias:	?>
/assignvariableop_2_batch_normalization_16_gamma:	?=
.assignvariableop_3_batch_normalization_16_beta:	?D
5assignvariableop_4_batch_normalization_16_moving_mean:	?H
9assignvariableop_5_batch_normalization_16_moving_variance:	??
#assignvariableop_6_conv2d_18_kernel:??0
!assignvariableop_7_conv2d_18_bias:	?>
/assignvariableop_8_batch_normalization_17_gamma:	?=
.assignvariableop_9_batch_normalization_17_beta:	?E
6assignvariableop_10_batch_normalization_17_moving_mean:	?I
:assignvariableop_11_batch_normalization_17_moving_variance:	?@
$assignvariableop_12_conv2d_19_kernel:??1
"assignvariableop_13_conv2d_19_bias:	??
0assignvariableop_14_batch_normalization_18_gamma:	?>
/assignvariableop_15_batch_normalization_18_beta:	?E
6assignvariableop_16_batch_normalization_18_moving_mean:	?I
:assignvariableop_17_batch_normalization_18_moving_variance:	??
$assignvariableop_18_conv2d_20_kernel:?@0
"assignvariableop_19_conv2d_20_bias:@>
0assignvariableop_20_batch_normalization_19_gamma:@=
/assignvariableop_21_batch_normalization_19_beta:@D
6assignvariableop_22_batch_normalization_19_moving_mean:@H
:assignvariableop_23_batch_normalization_19_moving_variance:@6
"assignvariableop_24_dense_4_kernel:
??.
 assignvariableop_25_dense_4_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: #
assignvariableop_31_total: #
assignvariableop_32_count: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: F
+assignvariableop_35_adam_conv2d_17_kernel_m:?8
)assignvariableop_36_adam_conv2d_17_bias_m:	?F
7assignvariableop_37_adam_batch_normalization_16_gamma_m:	?E
6assignvariableop_38_adam_batch_normalization_16_beta_m:	?G
+assignvariableop_39_adam_conv2d_18_kernel_m:??8
)assignvariableop_40_adam_conv2d_18_bias_m:	?F
7assignvariableop_41_adam_batch_normalization_17_gamma_m:	?E
6assignvariableop_42_adam_batch_normalization_17_beta_m:	?G
+assignvariableop_43_adam_conv2d_19_kernel_m:??8
)assignvariableop_44_adam_conv2d_19_bias_m:	?F
7assignvariableop_45_adam_batch_normalization_18_gamma_m:	?E
6assignvariableop_46_adam_batch_normalization_18_beta_m:	?F
+assignvariableop_47_adam_conv2d_20_kernel_m:?@7
)assignvariableop_48_adam_conv2d_20_bias_m:@E
7assignvariableop_49_adam_batch_normalization_19_gamma_m:@D
6assignvariableop_50_adam_batch_normalization_19_beta_m:@=
)assignvariableop_51_adam_dense_4_kernel_m:
??5
'assignvariableop_52_adam_dense_4_bias_m:F
+assignvariableop_53_adam_conv2d_17_kernel_v:?8
)assignvariableop_54_adam_conv2d_17_bias_v:	?F
7assignvariableop_55_adam_batch_normalization_16_gamma_v:	?E
6assignvariableop_56_adam_batch_normalization_16_beta_v:	?G
+assignvariableop_57_adam_conv2d_18_kernel_v:??8
)assignvariableop_58_adam_conv2d_18_bias_v:	?F
7assignvariableop_59_adam_batch_normalization_17_gamma_v:	?E
6assignvariableop_60_adam_batch_normalization_17_beta_v:	?G
+assignvariableop_61_adam_conv2d_19_kernel_v:??8
)assignvariableop_62_adam_conv2d_19_bias_v:	?F
7assignvariableop_63_adam_batch_normalization_18_gamma_v:	?E
6assignvariableop_64_adam_batch_normalization_18_beta_v:	?F
+assignvariableop_65_adam_conv2d_20_kernel_v:?@7
)assignvariableop_66_adam_conv2d_20_bias_v:@E
7assignvariableop_67_adam_batch_normalization_19_gamma_v:@D
6assignvariableop_68_adam_batch_normalization_19_beta_v:@=
)assignvariableop_69_adam_dense_4_kernel_v:
??5
'assignvariableop_70_adam_dense_4_bias_v:
identity_72??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?&
value?&B?&HB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*?
value?B?HB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_17_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_17_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_16_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_16_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_16_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_16_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_18_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_18_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_17_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_17_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_17_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_17_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_19_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_19_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_18_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_18_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_18_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_18_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_20_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_20_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_19_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_19_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_19_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_19_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_17_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_17_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_16_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_16_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_18_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_18_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp7assignvariableop_41_adam_batch_normalization_17_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_17_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_19_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_19_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_18_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_18_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_20_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_20_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_19_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_19_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_4_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adam_dense_4_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_17_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_17_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_batch_normalization_16_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adam_batch_normalization_16_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_18_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_18_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_17_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_17_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_19_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_19_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_batch_normalization_18_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adam_batch_normalization_18_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_20_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_20_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_19_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_19_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_dense_4_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_dense_4_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_72IdentityIdentity_71:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_72Identity_72:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18917

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@??
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?

c
D__inference_dropout_4_layer_call_and_return_conditional_losses_17461

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????q
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????k
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_16933

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19070

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  @?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
,__inference_sequential_5_layer_call_fn_17431
conv2d_17_input"
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?
	unknown_4:	?%
	unknown_5:??
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?

unknown_10:	?&

unknown_11:??

unknown_12:	?

unknown_13:	?

unknown_14:	?

unknown_15:	?

unknown_16:	?%

unknown_17:?@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:
??

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_17376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?
?
6__inference_batch_normalization_17_layer_call_fn_18697

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17233z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?J
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_17376

inputs*
conv2d_17_17154:?
conv2d_17_17156:	?+
batch_normalization_16_17184:	?+
batch_normalization_16_17186:	?+
batch_normalization_16_17188:	?+
batch_normalization_16_17190:	?+
conv2d_18_17204:??
conv2d_18_17206:	?+
batch_normalization_17_17234:	?+
batch_normalization_17_17236:	?+
batch_normalization_17_17238:	?+
batch_normalization_17_17240:	?+
conv2d_19_17254:??
conv2d_19_17256:	?+
batch_normalization_18_17284:	?+
batch_normalization_18_17286:	?+
batch_normalization_18_17288:	?+
batch_normalization_18_17290:	?*
conv2d_20_17304:?@
conv2d_20_17306:@*
batch_normalization_19_17334:@*
batch_normalization_19_17336:@*
batch_normalization_19_17338:@*
batch_normalization_19_17340:@!
dense_4_17370:
??
dense_4_17372:
identity??.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_17154conv2d_17_17156*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153?
leaky_re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0batch_normalization_16_17184batch_normalization_16_17186batch_normalization_16_17188batch_normalization_16_17190*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17183?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_18_17204conv2d_18_17206*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203?
leaky_re_lu_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0batch_normalization_17_17234batch_normalization_17_17236batch_normalization_17_17238batch_normalization_17_17240*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17233?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_19_17254conv2d_19_17256*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253?
leaky_re_lu_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0batch_normalization_18_17284batch_normalization_18_17286batch_normalization_18_17288batch_normalization_18_17290*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17283?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_20_17304conv2d_20_17306*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303?
leaky_re_lu_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0batch_normalization_19_17334batch_normalization_19_17336batch_normalization_19_17338batch_normalization_19_17340*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17333?
flatten_4/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349?
dropout_4/PartitionedCallPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17356?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_4_17370dense_4_17372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_17369w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?K
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18094
conv2d_17_input*
conv2d_17_18026:?
conv2d_17_18028:	?+
batch_normalization_16_18032:	?+
batch_normalization_16_18034:	?+
batch_normalization_16_18036:	?+
batch_normalization_16_18038:	?+
conv2d_18_18041:??
conv2d_18_18043:	?+
batch_normalization_17_18047:	?+
batch_normalization_17_18049:	?+
batch_normalization_17_18051:	?+
batch_normalization_17_18053:	?+
conv2d_19_18056:??
conv2d_19_18058:	?+
batch_normalization_18_18062:	?+
batch_normalization_18_18064:	?+
batch_normalization_18_18066:	?+
batch_normalization_18_18068:	?*
conv2d_20_18071:?@
conv2d_20_18073:@*
batch_normalization_19_18077:@*
batch_normalization_19_18079:@*
batch_normalization_19_18081:@*
batch_normalization_19_18083:@!
dense_4_18088:
??
dense_4_18090:
identity??.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallconv2d_17_inputconv2d_17_18026conv2d_17_18028*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153?
leaky_re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0batch_normalization_16_18032batch_normalization_16_18034batch_normalization_16_18036batch_normalization_16_18038*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17683?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_18_18041conv2d_18_18043*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203?
leaky_re_lu_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0batch_normalization_17_18047batch_normalization_17_18049batch_normalization_17_18051batch_normalization_17_18053*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17623?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_19_18056conv2d_19_18058*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253?
leaky_re_lu_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0batch_normalization_18_18062batch_normalization_18_18064batch_normalization_18_18066batch_normalization_18_18068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17563?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_20_18071conv2d_20_18073*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303?
leaky_re_lu_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0batch_normalization_19_18077batch_normalization_19_18079batch_normalization_19_18081batch_normalization_19_18083*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17503?
flatten_4/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17461?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_18088dense_4_18090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_17369w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_17_input
?
?
6__inference_batch_normalization_17_layer_call_fn_18710

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17623z
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*2
_output_shapes 
:????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17061

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
)__inference_dropout_4_layer_call_fn_19109

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17461q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_16966

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18476

inputsC
(conv2d_17_conv2d_readvariableop_resource:?8
)conv2d_17_biasadd_readvariableop_resource:	?=
.batch_normalization_16_readvariableop_resource:	??
0batch_normalization_16_readvariableop_1_resource:	?N
?batch_normalization_16_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_18_conv2d_readvariableop_resource:??8
)conv2d_18_biasadd_readvariableop_resource:	?=
.batch_normalization_17_readvariableop_resource:	??
0batch_normalization_17_readvariableop_1_resource:	?N
?batch_normalization_17_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_19_conv2d_readvariableop_resource:??8
)conv2d_19_biasadd_readvariableop_resource:	?=
.batch_normalization_18_readvariableop_resource:	??
0batch_normalization_18_readvariableop_1_resource:	?N
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:	?C
(conv2d_20_conv2d_readvariableop_resource:?@7
)conv2d_20_biasadd_readvariableop_resource:@<
.batch_normalization_19_readvariableop_resource:@>
0batch_normalization_19_readvariableop_1_resource:@M
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:@:
&dense_4_matmul_readvariableop_resource:
??5
'dense_4_biasadd_readvariableop_resource:
identity??%batch_normalization_16/AssignNewValue?'batch_normalization_16/AssignNewValue_1?6batch_normalization_16/FusedBatchNormV3/ReadVariableOp?8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_16/ReadVariableOp?'batch_normalization_16/ReadVariableOp_1?%batch_normalization_17/AssignNewValue?'batch_normalization_17/AssignNewValue_1?6batch_normalization_17/FusedBatchNormV3/ReadVariableOp?8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_17/ReadVariableOp?'batch_normalization_17/ReadVariableOp_1?%batch_normalization_18/AssignNewValue?'batch_normalization_18/AssignNewValue_1?6batch_normalization_18/FusedBatchNormV3/ReadVariableOp?8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_18/ReadVariableOp?'batch_normalization_18/ReadVariableOp_1?%batch_normalization_19/AssignNewValue?'batch_normalization_19/AssignNewValue_1?6batch_normalization_19/FusedBatchNormV3/ReadVariableOp?8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_19/ReadVariableOp?'batch_normalization_19/ReadVariableOp_1? conv2d_17/BiasAdd/ReadVariableOp?conv2d_17/Conv2D/ReadVariableOp? conv2d_18/BiasAdd/ReadVariableOp?conv2d_18/Conv2D/ReadVariableOp? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
conv2d_17/Conv2D/ReadVariableOpReadVariableOp(conv2d_17_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_17/Conv2DConv2Dinputs'conv2d_17/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
 conv2d_17/BiasAdd/ReadVariableOpReadVariableOp)conv2d_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_17/BiasAddBiasAddconv2d_17/Conv2D:output:0(conv2d_17/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
leaky_re_lu_17/LeakyRelu	LeakyReluconv2d_17/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
%batch_normalization_16/ReadVariableOpReadVariableOp.batch_normalization_16_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_16/ReadVariableOp_1ReadVariableOp0batch_normalization_16_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_16/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_16/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_17/LeakyRelu:activations:0-batch_normalization_16/ReadVariableOp:value:0/batch_normalization_16/ReadVariableOp_1:value:0>batch_normalization_16/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_16/AssignNewValueAssignVariableOp?batch_normalization_16_fusedbatchnormv3_readvariableop_resource4batch_normalization_16/FusedBatchNormV3:batch_mean:07^batch_normalization_16/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_16/AssignNewValue_1AssignVariableOpAbatch_normalization_16_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_16/FusedBatchNormV3:batch_variance:09^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_18/Conv2DConv2D+batch_normalization_16/FusedBatchNormV3:y:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*2
_output_shapes 
:????????????*
paddingSAME*
strides
?
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*2
_output_shapes 
:?????????????
leaky_re_lu_18/LeakyRelu	LeakyReluconv2d_18/BiasAdd:output:0*2
_output_shapes 
:????????????*
alpha%
?#<?
%batch_normalization_17/ReadVariableOpReadVariableOp.batch_normalization_17_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_17/ReadVariableOp_1ReadVariableOp0batch_normalization_17_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_17/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_17/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_18/LeakyRelu:activations:0-batch_normalization_17/ReadVariableOp:value:0/batch_normalization_17/ReadVariableOp_1:value:0>batch_normalization_17/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_17/AssignNewValueAssignVariableOp?batch_normalization_17_fusedbatchnormv3_readvariableop_resource4batch_normalization_17/FusedBatchNormV3:batch_mean:07^batch_normalization_17/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_17/AssignNewValue_1AssignVariableOpAbatch_normalization_17_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_17/FusedBatchNormV3:batch_variance:09^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_19/Conv2DConv2D+batch_normalization_17/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@??
leaky_re_lu_19/LeakyRelu	LeakyReluconv2d_19/BiasAdd:output:0*0
_output_shapes
:?????????@@?*
alpha%
?#<?
%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_19/LeakyRelu:activations:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv2d_20/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @*
paddingSAME*
strides
?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  @?
leaky_re_lu_20/LeakyRelu	LeakyReluconv2d_20/BiasAdd:output:0*/
_output_shapes
:?????????  @*
alpha%
?#<?
%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_20/LeakyRelu:activations:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  @:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_4/ReshapeReshape+batch_normalization_19/FusedBatchNormV3:y:0flatten_4/Const:output:0*
T0*)
_output_shapes
:???????????\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
dropout_4/dropout/MulMulflatten_4/Reshape:output:0 dropout_4/dropout/Const:output:0*
T0*)
_output_shapes
:???????????a
dropout_4/dropout/ShapeShapeflatten_4/Reshape:output:0*
T0*
_output_shapes
:?
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:????????????
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:????????????
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*)
_output_shapes
:????????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_4/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^batch_normalization_16/AssignNewValue(^batch_normalization_16/AssignNewValue_17^batch_normalization_16/FusedBatchNormV3/ReadVariableOp9^batch_normalization_16/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_16/ReadVariableOp(^batch_normalization_16/ReadVariableOp_1&^batch_normalization_17/AssignNewValue(^batch_normalization_17/AssignNewValue_17^batch_normalization_17/FusedBatchNormV3/ReadVariableOp9^batch_normalization_17/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_17/ReadVariableOp(^batch_normalization_17/ReadVariableOp_1&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1!^conv2d_17/BiasAdd/ReadVariableOp ^conv2d_17/Conv2D/ReadVariableOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_16/AssignNewValue%batch_normalization_16/AssignNewValue2R
'batch_normalization_16/AssignNewValue_1'batch_normalization_16/AssignNewValue_12p
6batch_normalization_16/FusedBatchNormV3/ReadVariableOp6batch_normalization_16/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_16/FusedBatchNormV3/ReadVariableOp_18batch_normalization_16/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_16/ReadVariableOp%batch_normalization_16/ReadVariableOp2R
'batch_normalization_16/ReadVariableOp_1'batch_normalization_16/ReadVariableOp_12N
%batch_normalization_17/AssignNewValue%batch_normalization_17/AssignNewValue2R
'batch_normalization_17/AssignNewValue_1'batch_normalization_17/AssignNewValue_12p
6batch_normalization_17/FusedBatchNormV3/ReadVariableOp6batch_normalization_17/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_17/FusedBatchNormV3/ReadVariableOp_18batch_normalization_17/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_17/ReadVariableOp%batch_normalization_17/ReadVariableOp2R
'batch_normalization_17/ReadVariableOp_1'batch_normalization_17/ReadVariableOp_12N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12D
 conv2d_17/BiasAdd/ReadVariableOp conv2d_17/BiasAdd/ReadVariableOp2B
conv2d_17/Conv2D/ReadVariableOpconv2d_17/Conv2D/ReadVariableOp2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?K
?
G__inference_sequential_5_layer_call_and_return_conditional_losses_17840

inputs*
conv2d_17_17772:?
conv2d_17_17774:	?+
batch_normalization_16_17778:	?+
batch_normalization_16_17780:	?+
batch_normalization_16_17782:	?+
batch_normalization_16_17784:	?+
conv2d_18_17787:??
conv2d_18_17789:	?+
batch_normalization_17_17793:	?+
batch_normalization_17_17795:	?+
batch_normalization_17_17797:	?+
batch_normalization_17_17799:	?+
conv2d_19_17802:??
conv2d_19_17804:	?+
batch_normalization_18_17808:	?+
batch_normalization_18_17810:	?+
batch_normalization_18_17812:	?+
batch_normalization_18_17814:	?*
conv2d_20_17817:?@
conv2d_20_17819:@*
batch_normalization_19_17823:@*
batch_normalization_19_17825:@*
batch_normalization_19_17827:@*
batch_normalization_19_17829:@!
dense_4_17834:
??
dense_4_17836:
identity??.batch_normalization_16/StatefulPartitionedCall?.batch_normalization_17/StatefulPartitionedCall?.batch_normalization_18/StatefulPartitionedCall?.batch_normalization_19/StatefulPartitionedCall?!conv2d_17/StatefulPartitionedCall?!conv2d_18/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?!dropout_4/StatefulPartitionedCall?
!conv2d_17/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_17_17772conv2d_17_17774*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_17_layer_call_and_return_conditional_losses_17153?
leaky_re_lu_17/PartitionedCallPartitionedCall*conv2d_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_17164?
.batch_normalization_16/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_17/PartitionedCall:output:0batch_normalization_16_17778batch_normalization_16_17780batch_normalization_16_17782batch_normalization_16_17784*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_17683?
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_16/StatefulPartitionedCall:output:0conv2d_18_17787conv2d_18_17789*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_17203?
leaky_re_lu_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_17214?
.batch_normalization_17/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_18/PartitionedCall:output:0batch_normalization_17_17793batch_normalization_17_17795batch_normalization_17_17797batch_normalization_17_17799*
Tin	
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_17623?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_17/StatefulPartitionedCall:output:0conv2d_19_17802conv2d_19_17804*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_17253?
leaky_re_lu_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_17264?
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_19/PartitionedCall:output:0batch_normalization_18_17808batch_normalization_18_17810batch_normalization_18_17812batch_normalization_18_17814*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_17563?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_20_17817conv2d_20_17819*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_17303?
leaky_re_lu_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_17314?
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0batch_normalization_19_17823batch_normalization_19_17825batch_normalization_19_17827batch_normalization_19_17829*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_17503?
flatten_4/PartitionedCallPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_17349?
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_17461?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_4_17834dense_4_17836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_17369w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_16/StatefulPartitionedCall/^batch_normalization_17/StatefulPartitionedCall/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall"^conv2d_17/StatefulPartitionedCall"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_16/StatefulPartitionedCall.batch_normalization_16/StatefulPartitionedCall2`
.batch_normalization_17/StatefulPartitionedCall.batch_normalization_17/StatefulPartitionedCall2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2F
!conv2d_17/StatefulPartitionedCall!conv2d_17/StatefulPartitionedCall2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18629

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*R
_output_shapes@
>:????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0n
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*2
_output_shapes 
:?????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Z V
2
_output_shapes 
:????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_17_inputB
!serving_default_conv2d_17_input:0???????????;
dense_40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/	variables
0trainable_variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Faxis
	Ggamma
Hbeta
Imoving_mean
Jmoving_variance
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Okernel
Pbias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?!m?"m?)m?*m?4m?5m?<m?=m?Gm?Hm?Om?Pm?Zm?[m?jm?km?v?v?!v?"v?)v?*v?4v?5v?<v?=v?Gv?Hv?Ov?Pv?Zv?[v?jv?kv?"
	optimizer
?
0
1
!2
"3
#4
$5
)6
*7
48
59
610
711
<12
=13
G14
H15
I16
J17
O18
P19
Z20
[21
\22
]23
j24
k25"
trackable_list_wrapper
?
0
1
!2
"3
)4
*5
46
57
<8
=9
G10
H11
O12
P13
Z14
[15
j16
k17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
+:)?2conv2d_17/kernel
:?2conv2d_17/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_16/gamma
*:(?2batch_normalization_16/beta
3:1? (2"batch_normalization_16/moving_mean
7:5? (2&batch_normalization_16/moving_variance
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_18/kernel
:?2conv2d_18/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
/	variables
0trainable_variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_17/gamma
*:(?2batch_normalization_17/beta
3:1? (2"batch_normalization_17/moving_mean
7:5? (2&batch_normalization_17/moving_variance
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
8	variables
9trainable_variables
:regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_19/kernel
:?2conv2d_19/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
>	variables
?trainable_variables
@regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)?2batch_normalization_18/gamma
*:(?2batch_normalization_18/beta
3:1? (2"batch_normalization_18/moving_mean
7:5? (2&batch_normalization_18/moving_variance
<
G0
H1
I2
J3"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_20/kernel
:@2conv2d_20/bias
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_19/gamma
):'@2batch_normalization_19/beta
2:0@ (2"batch_normalization_19/moving_mean
6:4@ (2&batch_normalization_19/moving_variance
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
f	variables
gtrainable_variables
hregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_4/kernel
:2dense_4/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
l	variables
mtrainable_variables
nregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
X
#0
$1
62
73
I4
J5
\6
]7"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
0:.?2Adam/conv2d_17/kernel/m
": ?2Adam/conv2d_17/bias/m
0:.?2#Adam/batch_normalization_16/gamma/m
/:-?2"Adam/batch_normalization_16/beta/m
1:/??2Adam/conv2d_18/kernel/m
": ?2Adam/conv2d_18/bias/m
0:.?2#Adam/batch_normalization_17/gamma/m
/:-?2"Adam/batch_normalization_17/beta/m
1:/??2Adam/conv2d_19/kernel/m
": ?2Adam/conv2d_19/bias/m
0:.?2#Adam/batch_normalization_18/gamma/m
/:-?2"Adam/batch_normalization_18/beta/m
0:.?@2Adam/conv2d_20/kernel/m
!:@2Adam/conv2d_20/bias/m
/:-@2#Adam/batch_normalization_19/gamma/m
.:,@2"Adam/batch_normalization_19/beta/m
':%
??2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
0:.?2Adam/conv2d_17/kernel/v
": ?2Adam/conv2d_17/bias/v
0:.?2#Adam/batch_normalization_16/gamma/v
/:-?2"Adam/batch_normalization_16/beta/v
1:/??2Adam/conv2d_18/kernel/v
": ?2Adam/conv2d_18/bias/v
0:.?2#Adam/batch_normalization_17/gamma/v
/:-?2"Adam/batch_normalization_17/beta/v
1:/??2Adam/conv2d_19/kernel/v
": ?2Adam/conv2d_19/bias/v
0:.?2#Adam/batch_normalization_18/gamma/v
/:-?2"Adam/batch_normalization_18/beta/v
0:.?@2Adam/conv2d_20/kernel/v
!:@2Adam/conv2d_20/bias/v
/:-@2#Adam/batch_normalization_19/gamma/v
.:,@2"Adam/batch_normalization_19/beta/v
':%
??2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
?2?
,__inference_sequential_5_layer_call_fn_17431
,__inference_sequential_5_layer_call_fn_18216
,__inference_sequential_5_layer_call_fn_18273
,__inference_sequential_5_layer_call_fn_17952?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18371
G__inference_sequential_5_layer_call_and_return_conditional_losses_18476
G__inference_sequential_5_layer_call_and_return_conditional_losses_18023
G__inference_sequential_5_layer_call_and_return_conditional_losses_18094?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_16880conv2d_17_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_17_layer_call_fn_18485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_17_layer_call_fn_18500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_18505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_16_layer_call_fn_18518
6__inference_batch_normalization_16_layer_call_fn_18531
6__inference_batch_normalization_16_layer_call_fn_18544
6__inference_batch_normalization_16_layer_call_fn_18557?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18575
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18593
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18611
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18629?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_18_layer_call_fn_18638?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_18_layer_call_and_return_conditional_losses_18648?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_18_layer_call_fn_18653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_18658?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_17_layer_call_fn_18671
6__inference_batch_normalization_17_layer_call_fn_18684
6__inference_batch_normalization_17_layer_call_fn_18697
6__inference_batch_normalization_17_layer_call_fn_18710?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18728
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18746
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18764
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18782?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_19_layer_call_fn_18791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_18801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_19_layer_call_fn_18806?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_18811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_18_layer_call_fn_18824
6__inference_batch_normalization_18_layer_call_fn_18837
6__inference_batch_normalization_18_layer_call_fn_18850
6__inference_batch_normalization_18_layer_call_fn_18863?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18881
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18899
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18917
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18935?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_20_layer_call_fn_18944?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_20_layer_call_fn_18959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_18964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_batch_normalization_19_layer_call_fn_18977
6__inference_batch_normalization_19_layer_call_fn_18990
6__inference_batch_normalization_19_layer_call_fn_19003
6__inference_batch_normalization_19_layer_call_fn_19016?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19034
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19052
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19070
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19088?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_flatten_4_layer_call_fn_19093?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_4_layer_call_and_return_conditional_losses_19099?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_4_layer_call_fn_19104
)__inference_dropout_4_layer_call_fn_19109?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_4_layer_call_and_return_conditional_losses_19114
D__inference_dropout_4_layer_call_and_return_conditional_losses_19126?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_19135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_19146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_18159conv2d_17_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_16880?!"#$)*4567<=GHIJOPZ[\]jkB??
8?5
3?0
conv2d_17_input???????????
? "1?.
,
dense_4!?
dense_4??????????
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18575?!"#$N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18593?!"#$N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18611x!"#$>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_16_layer_call_and_return_conditional_losses_18629x!"#$>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
6__inference_batch_normalization_16_layer_call_fn_18518?!"#$N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_16_layer_call_fn_18531?!"#$N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_16_layer_call_fn_18544k!"#$>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
6__inference_batch_normalization_16_layer_call_fn_18557k!"#$>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18728?4567N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18746?4567N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18764x4567>?;
4?1
+?(
inputs????????????
p 
? "0?-
&?#
0????????????
? ?
Q__inference_batch_normalization_17_layer_call_and_return_conditional_losses_18782x4567>?;
4?1
+?(
inputs????????????
p
? "0?-
&?#
0????????????
? ?
6__inference_batch_normalization_17_layer_call_fn_18671?4567N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_17_layer_call_fn_18684?4567N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_17_layer_call_fn_18697k4567>?;
4?1
+?(
inputs????????????
p 
? "#? ?????????????
6__inference_batch_normalization_17_layer_call_fn_18710k4567>?;
4?1
+?(
inputs????????????
p
? "#? ?????????????
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18881?GHIJN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18899?GHIJN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18917tGHIJ<?9
2?/
)?&
inputs?????????@@?
p 
? ".?+
$?!
0?????????@@?
? ?
Q__inference_batch_normalization_18_layer_call_and_return_conditional_losses_18935tGHIJ<?9
2?/
)?&
inputs?????????@@?
p
? ".?+
$?!
0?????????@@?
? ?
6__inference_batch_normalization_18_layer_call_fn_18824?GHIJN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_18_layer_call_fn_18837?GHIJN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
6__inference_batch_normalization_18_layer_call_fn_18850gGHIJ<?9
2?/
)?&
inputs?????????@@?
p 
? "!??????????@@??
6__inference_batch_normalization_18_layer_call_fn_18863gGHIJ<?9
2?/
)?&
inputs?????????@@?
p
? "!??????????@@??
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19034?Z[\]M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19052?Z[\]M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19070rZ[\];?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
Q__inference_batch_normalization_19_layer_call_and_return_conditional_losses_19088rZ[\];?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
6__inference_batch_normalization_19_layer_call_fn_18977?Z[\]M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_19_layer_call_fn_18990?Z[\]M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
6__inference_batch_normalization_19_layer_call_fn_19003eZ[\];?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
6__inference_batch_normalization_19_layer_call_fn_19016eZ[\];?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
D__inference_conv2d_17_layer_call_and_return_conditional_losses_18495q9?6
/?,
*?'
inputs???????????
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_17_layer_call_fn_18485d9?6
/?,
*?'
inputs???????????
? "#? ?????????????
D__inference_conv2d_18_layer_call_and_return_conditional_losses_18648r)*:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
)__inference_conv2d_18_layer_call_fn_18638e)*:?7
0?-
+?(
inputs????????????
? "#? ?????????????
D__inference_conv2d_19_layer_call_and_return_conditional_losses_18801p<=:?7
0?-
+?(
inputs????????????
? ".?+
$?!
0?????????@@?
? ?
)__inference_conv2d_19_layer_call_fn_18791c<=:?7
0?-
+?(
inputs????????????
? "!??????????@@??
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18954mOP8?5
.?+
)?&
inputs?????????@@?
? "-?*
#? 
0?????????  @
? ?
)__inference_conv2d_20_layer_call_fn_18944`OP8?5
.?+
)?&
inputs?????????@@?
? " ??????????  @?
B__inference_dense_4_layer_call_and_return_conditional_losses_19146^jk1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? |
'__inference_dense_4_layer_call_fn_19135Qjk1?.
'?$
"?
inputs???????????
? "???????????
D__inference_dropout_4_layer_call_and_return_conditional_losses_19114`5?2
+?(
"?
inputs???????????
p 
? "'?$
?
0???????????
? ?
D__inference_dropout_4_layer_call_and_return_conditional_losses_19126`5?2
+?(
"?
inputs???????????
p
? "'?$
?
0???????????
? ?
)__inference_dropout_4_layer_call_fn_19104S5?2
+?(
"?
inputs???????????
p 
? "?????????????
)__inference_dropout_4_layer_call_fn_19109S5?2
+?(
"?
inputs???????????
p
? "?????????????
D__inference_flatten_4_layer_call_and_return_conditional_losses_19099b7?4
-?*
(?%
inputs?????????  @
? "'?$
?
0???????????
? ?
)__inference_flatten_4_layer_call_fn_19093U7?4
-?*
(?%
inputs?????????  @
? "?????????????
I__inference_leaky_re_lu_17_layer_call_and_return_conditional_losses_18505n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
.__inference_leaky_re_lu_17_layer_call_fn_18500a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
I__inference_leaky_re_lu_18_layer_call_and_return_conditional_losses_18658n:?7
0?-
+?(
inputs????????????
? "0?-
&?#
0????????????
? ?
.__inference_leaky_re_lu_18_layer_call_fn_18653a:?7
0?-
+?(
inputs????????????
? "#? ?????????????
I__inference_leaky_re_lu_19_layer_call_and_return_conditional_losses_18811j8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????@@?
? ?
.__inference_leaky_re_lu_19_layer_call_fn_18806]8?5
.?+
)?&
inputs?????????@@?
? "!??????????@@??
I__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_18964h7?4
-?*
(?%
inputs?????????  @
? "-?*
#? 
0?????????  @
? ?
.__inference_leaky_re_lu_20_layer_call_fn_18959[7?4
-?*
(?%
inputs?????????  @
? " ??????????  @?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18023?!"#$)*4567<=GHIJOPZ[\]jkJ?G
@?=
3?0
conv2d_17_input???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18094?!"#$)*4567<=GHIJOPZ[\]jkJ?G
@?=
3?0
conv2d_17_input???????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18371?!"#$)*4567<=GHIJOPZ[\]jkA?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_5_layer_call_and_return_conditional_losses_18476?!"#$)*4567<=GHIJOPZ[\]jkA?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_5_layer_call_fn_17431?!"#$)*4567<=GHIJOPZ[\]jkJ?G
@?=
3?0
conv2d_17_input???????????
p 

 
? "???????????
,__inference_sequential_5_layer_call_fn_17952?!"#$)*4567<=GHIJOPZ[\]jkJ?G
@?=
3?0
conv2d_17_input???????????
p

 
? "???????????
,__inference_sequential_5_layer_call_fn_18216y!"#$)*4567<=GHIJOPZ[\]jkA?>
7?4
*?'
inputs???????????
p 

 
? "???????????
,__inference_sequential_5_layer_call_fn_18273y!"#$)*4567<=GHIJOPZ[\]jkA?>
7?4
*?'
inputs???????????
p

 
? "???????????
#__inference_signature_wrapper_18159?!"#$)*4567<=GHIJOPZ[\]jkU?R
? 
K?H
F
conv2d_17_input3?0
conv2d_17_input???????????"1?.
,
dense_4!?
dense_4?????????