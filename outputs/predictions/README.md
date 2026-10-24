# Out-of-sample predictions

Store validation and final-test predictions for the baseline, AR, Ridge, and LightGBM. The fields for this legacy folder are timestamp,target_end,model,y_true,y_pred,split. Both y_true and y_pred are fifteen-minute log returns, and times use UTC. Use the same evaluation dates and investigate missing predictions first.
