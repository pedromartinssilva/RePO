до
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8║с
И
my_model/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
─А*&
shared_namemy_model/dense/kernel
Б
)my_model/dense/kernel/Read/ReadVariableOpReadVariableOpmy_model/dense/kernel* 
_output_shapes
:
─А*
dtype0

my_model/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_namemy_model/dense/bias
x
'my_model/dense/bias/Read/ReadVariableOpReadVariableOpmy_model/dense/bias*
_output_shapes	
:А*
dtype0
М
my_model/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А─*(
shared_namemy_model/dense_1/kernel
Е
+my_model/dense_1/kernel/Read/ReadVariableOpReadVariableOpmy_model/dense_1/kernel* 
_output_shapes
:
А─*
dtype0
Г
my_model/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:─*&
shared_namemy_model/dense_1/bias
|
)my_model/dense_1/bias/Read/ReadVariableOpReadVariableOpmy_model/dense_1/bias*
_output_shapes	
:─*
dtype0

NoOpNoOp
ї
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*░
valueжBг BЬ

flatten
d1
d2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
R
	trainable_variables

	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
Ъ
metrics
non_trainable_variables

layers
layer_regularization_losses
trainable_variables
	variables
regularization_losses
 
 
 
 
Ъ
metrics
non_trainable_variables

layers
 layer_regularization_losses
	trainable_variables

	variables
regularization_losses
OM
VARIABLE_VALUEmy_model/dense/kernel$d1/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEmy_model/dense/bias"d1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
!metrics
"non_trainable_variables

#layers
$layer_regularization_losses
trainable_variables
	variables
regularization_losses
QO
VARIABLE_VALUEmy_model/dense_1/kernel$d2/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEmy_model/dense_1/bias"d2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
%metrics
&non_trainable_variables

'layers
(layer_regularization_losses
trainable_variables
	variables
regularization_losses
 
 

0
1
2
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
В
serving_default_input_1Placeholder*+
_output_shapes
:         *
dtype0* 
shape:         
■
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model/dense/kernelmy_model/dense/biasmy_model/dense_1/kernelmy_model/dense_1/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference_signature_wrapper_517
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╡
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)my_model/dense/kernel/Read/ReadVariableOp'my_model/dense/bias/Read/ReadVariableOp+my_model/dense_1/kernel/Read/ReadVariableOp)my_model/dense_1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*%
f R
__inference__traced_save_600
р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model/dense/kernelmy_model/dense/biasmy_model/dense_1/kernelmy_model/dense_1/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_restore_624Т┴
Г
\
@__inference_flatten_layer_call_and_return_conditional_losses_523

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    D  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ─2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0**
_input_shapes
:         :& "
 
_user_specified_nameinputs
Ё
ж
%__inference_dense_1_layer_call_fn_564

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ─*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_4772
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╔	
┘
@__inference_dense_1_layer_call_and_return_conditional_losses_557

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А─*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:─*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ─2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╘
A
%__inference_flatten_layer_call_fn_528

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ─*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_4332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0**
_input_shapes
:         :& "
 
_user_specified_nameinputs
╟	
╫
>__inference_dense_layer_call_and_return_conditional_losses_453

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
─А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ─::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Р
Ё
&__inference_my_model_layer_call_fn_507
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_my_model_layer_call_and_return_conditional_losses_4972
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
ь
д
#__inference_dense_layer_call_fn_546

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_4532
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ─::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
▐
╫
__inference__traced_save_600
file_prefix4
0savev2_my_model_dense_kernel_read_readvariableop2
.savev2_my_model_dense_bias_read_readvariableop6
2savev2_my_model_dense_1_kernel_read_readvariableop4
0savev2_my_model_dense_1_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_43d45c2a7f8d42ec9d5c6f44a801a483/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЧ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slicesў
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_my_model_dense_kernel_read_readvariableop.savev2_my_model_dense_bias_read_readvariableop2savev2_my_model_dense_1_kernel_read_readvariableop0savev2_my_model_dense_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*=
_input_shapes,
*: :
─А:А:
А─:─: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
о
╧
A__inference_my_model_layer_call_and_return_conditional_losses_497
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCall╜
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ─*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_4332
flatten/PartitionedCallЕ
flatten/IdentityIdentity flatten/PartitionedCall:output:0*
T0*(
_output_shapes
:         ─2
flatten/Identityп
dense/StatefulPartitionedCallStatefulPartitionedCallflatten/Identity:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         А*-
config_proto

GPU

CPU2*0J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_4532
dense/StatefulPartitionedCallз
dense/IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*(
_output_shapes
:         А2
dense/Identity╖
dense_1/StatefulPartitionedCallStatefulPartitionedCalldense/Identity:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         ─*-
config_proto

GPU

CPU2*0J 8*I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_4772!
dense_1/StatefulPartitionedCallп
dense_1/IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:         ─2
dense_1/Identityw
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
clip_by_value/Minimum/yй
clip_by_value/MinimumMinimumdense_1/Identity:output:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         ─2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/yС
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:         ─2
clip_by_values
Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Reshape/shape~
ReshapeReshapeclip_by_value:z:0Reshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeк
IdentityIdentityReshape:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ш
ы
!__inference_signature_wrapper_517
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:         *-
config_proto

GPU

CPU2*0J 8*'
f"R 
__inference__wrapped_model_4232
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Г
\
@__inference_flatten_layer_call_and_return_conditional_losses_433

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    D  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ─2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0**
_input_shapes
:         :& "
 
_user_specified_nameinputs
Э!
▓
__inference__wrapped_model_423
input_11
-my_model_dense_matmul_readvariableop_resource2
.my_model_dense_biasadd_readvariableop_resource3
/my_model_dense_1_matmul_readvariableop_resource4
0my_model_dense_1_biasadd_readvariableop_resource
identityИв%my_model/dense/BiasAdd/ReadVariableOpв$my_model/dense/MatMul/ReadVariableOpв'my_model/dense_1/BiasAdd/ReadVariableOpв&my_model/dense_1/MatMul/ReadVariableOpБ
my_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    D  2
my_model/flatten/ConstЬ
my_model/flatten/ReshapeReshapeinput_1my_model/flatten/Const:output:0*
T0*(
_output_shapes
:         ─2
my_model/flatten/Reshape╝
$my_model/dense/MatMul/ReadVariableOpReadVariableOp-my_model_dense_matmul_readvariableop_resource* 
_output_shapes
:
─А*
dtype02&
$my_model/dense/MatMul/ReadVariableOp╝
my_model/dense/MatMulMatMul!my_model/flatten/Reshape:output:0,my_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
my_model/dense/MatMul║
%my_model/dense/BiasAdd/ReadVariableOpReadVariableOp.my_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02'
%my_model/dense/BiasAdd/ReadVariableOp╛
my_model/dense/BiasAddBiasAddmy_model/dense/MatMul:product:0-my_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
my_model/dense/BiasAddЖ
my_model/dense/ReluRelumy_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
my_model/dense/Relu┬
&my_model/dense_1/MatMul/ReadVariableOpReadVariableOp/my_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
А─*
dtype02(
&my_model/dense_1/MatMul/ReadVariableOp┬
my_model/dense_1/MatMulMatMul!my_model/dense/Relu:activations:0.my_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2
my_model/dense_1/MatMul└
'my_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp0my_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:─*
dtype02)
'my_model/dense_1/BiasAdd/ReadVariableOp╞
my_model/dense_1/BiasAddBiasAdd!my_model/dense_1/MatMul:product:0/my_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2
my_model/dense_1/BiasAddМ
my_model/dense_1/ReluRelu!my_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ─2
my_model/dense_1/ReluЙ
 my_model/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2"
 my_model/clip_by_value/Minimum/y╬
my_model/clip_by_value/MinimumMinimum#my_model/dense_1/Relu:activations:0)my_model/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:         ─2 
my_model/clip_by_value/Minimumy
my_model/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
my_model/clip_by_value/y╡
my_model/clip_by_valueMaximum"my_model/clip_by_value/Minimum:z:0!my_model/clip_by_value/y:output:0*
T0*(
_output_shapes
:         ─2
my_model/clip_by_valueЕ
my_model/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"          2
my_model/Reshape/shapeв
my_model/ReshapeReshapemy_model/clip_by_value:z:0my_model/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
my_model/ReshapeУ
IdentityIdentitymy_model/Reshape:output:0&^my_model/dense/BiasAdd/ReadVariableOp%^my_model/dense/MatMul/ReadVariableOp(^my_model/dense_1/BiasAdd/ReadVariableOp'^my_model/dense_1/MatMul/ReadVariableOp*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         ::::2N
%my_model/dense/BiasAdd/ReadVariableOp%my_model/dense/BiasAdd/ReadVariableOp2L
$my_model/dense/MatMul/ReadVariableOp$my_model/dense/MatMul/ReadVariableOp2R
'my_model/dense_1/BiasAdd/ReadVariableOp'my_model/dense_1/BiasAdd/ReadVariableOp2P
&my_model/dense_1/MatMul/ReadVariableOp&my_model/dense_1/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
╟	
╫
>__inference_dense_layer_call_and_return_conditional_losses_539

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
─А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         А2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ─::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╔	
┘
@__inference_dense_1_layer_call_and_return_conditional_losses_477

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А─*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:─*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ─2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ─2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ─2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ю
ч
__inference__traced_restore_624
file_prefix*
&assignvariableop_my_model_dense_kernel*
&assignvariableop_1_my_model_dense_bias.
*assignvariableop_2_my_model_dense_1_kernel,
(assignvariableop_3_my_model_dense_1_bias

identity_5ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3в	RestoreV2вRestoreV2_1Э
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB$d1/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d1/bias/.ATTRIBUTES/VARIABLE_VALUEB$d2/kernel/.ATTRIBUTES/VARIABLE_VALUEB"d2/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices┐
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЦ
AssignVariableOpAssignVariableOp&assignvariableop_my_model_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ь
AssignVariableOp_1AssignVariableOp&assignvariableop_1_my_model_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2а
AssignVariableOp_2AssignVariableOp*assignvariableop_2_my_model_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ю
AssignVariableOp_3AssignVariableOp(assignvariableop_3_my_model_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4╞

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*│
serving_defaultЯ
?
input_14
serving_default_input_1:0         @
output_14
StatefulPartitionedCall:0         tensorflow/serving/predict:тE
¤
flatten
d1
d2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
)__call__
**&call_and_return_all_conditional_losses
+_default_save_signature"д
_tf_keras_modelК{"class_name": "MyModel", "name": "my_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}}
м
	trainable_variables

	variables
regularization_losses
	keras_api
,__call__
*-&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ё

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
.__call__
*/&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2048, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 580}}}}
Ї

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
0__call__
*1&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 580, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
╖
metrics
non_trainable_variables

layers
layer_regularization_losses
trainable_variables
	variables
regularization_losses
)__call__
+_default_save_signature
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
,
2serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
metrics
non_trainable_variables

layers
 layer_regularization_losses
	trainable_variables

	variables
regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
):'
─А2my_model/dense/kernel
": А2my_model/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
!metrics
"non_trainable_variables

#layers
$layer_regularization_losses
trainable_variables
	variables
regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
+:)
А─2my_model/dense_1/kernel
$:"─2my_model/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
%metrics
&non_trainable_variables

'layers
(layer_regularization_losses
trainable_variables
	variables
regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
є2Ё
&__inference_my_model_layer_call_fn_507┼
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
input_1         
О2Л
A__inference_my_model_layer_call_and_return_conditional_losses_497┼
Ф▓Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
input_1         
р2▌
__inference__wrapped_model_423║
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк **в'
%К"
input_1         
╧2╠
%__inference_flatten_layer_call_fn_528в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_flatten_layer_call_and_return_conditional_losses_523в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
═2╩
#__inference_dense_layer_call_fn_546в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ш2х
>__inference_dense_layer_call_and_return_conditional_losses_539в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
%__inference_dense_1_layer_call_fn_564в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ъ2ч
@__inference_dense_1_layer_call_and_return_conditional_losses_557в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0B.
!__inference_signature_wrapper_517input_1Ч
__inference__wrapped_model_423u4в1
*в'
%К"
input_1         
к "7к4
2
output_1&К#
output_1         в
@__inference_dense_1_layer_call_and_return_conditional_losses_557^0в-
&в#
!К
inputs         А
к "&в#
К
0         ─
Ъ z
%__inference_dense_1_layer_call_fn_564Q0в-
&в#
!К
inputs         А
к "К         ─а
>__inference_dense_layer_call_and_return_conditional_losses_539^0в-
&в#
!К
inputs         ─
к "&в#
К
0         А
Ъ x
#__inference_dense_layer_call_fn_546Q0в-
&в#
!К
inputs         ─
к "К         Аб
@__inference_flatten_layer_call_and_return_conditional_losses_523]3в0
)в&
$К!
inputs         
к "&в#
К
0         ─
Ъ y
%__inference_flatten_layer_call_fn_528P3в0
)в&
$К!
inputs         
к "К         ─м
A__inference_my_model_layer_call_and_return_conditional_losses_497g4в1
*в'
%К"
input_1         
к ")в&
К
0         
Ъ Д
&__inference_my_model_layer_call_fn_507Z4в1
*в'
%К"
input_1         
к "К         ж
!__inference_signature_wrapper_517А?в<
в 
5к2
0
input_1%К"
input_1         "7к4
2
output_1&К#
output_1         