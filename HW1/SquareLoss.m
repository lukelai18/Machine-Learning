function L = SquareLoss(Fx,y)
    % take in Fx and y, check if dimension matches
    % return square loss
    if length(Fx) ~= length(y)
        L=-1;
        return;
    end

    L=0;
    N = length(Fx)
    for i=1:N
        L=L+power(Fx(i)-y(i),2);
    end
    L = L/N;
end

