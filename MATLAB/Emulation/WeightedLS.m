function Beta = WeightedLS(D,W,y)
Beta = (D' * W * D) \ (D' * W * y);
end

