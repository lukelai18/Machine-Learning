function Y = Predictor(W, X, d)
    % Takes W as parameter, X input and d as model degree
    n=length(X);
    Y=zeros(n,1);
    for k=1:n
        y_k=0;
        for i=1:d+1
            y_k = y_k + power(X(k), i-1)*W(i);
        end
        Y(k)=y_k;
    end
end

