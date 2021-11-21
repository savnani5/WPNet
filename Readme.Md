UPDATE: train.py file is not tested, might contain bugs !

-- Have a baseline pytorch model for depth >> Done  

-- System Design and Boilerplate Code >> WIP
	
		a) Image load function >> Done

-- Write code for WPNet >> WIP

		a) Base Architecture code  >> Done
		b) Dataloader >> Done
		c) Loss function >> Done
		d) Train script >> Done (testing is remaining)
		e) Test script >> WIP
		e) Add device == cuda to the tensors >> Remaining

-- Train steps
	First try with depth model freezed
	Then try with few layers unfreezed 

-- Try Image stacking -- multiple frames

-- Depth + Semantic ??

-- Frame stacking wrapper - how to stack the frames ?

-- Also can stack depth + rgb to make 4 channel ?? Checkout 


Monodepth caveat - DONOT train from scratch - we do not have the compute capacity and time for that ! 
