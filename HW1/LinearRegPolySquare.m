function W = LinearRegPolySquare(X,Y,d)
    % takes X,Y training data, and d as model degree
    % return optimized W

    m=d+1; % num of w
    n = length(X) % size of data set
    W=zeros(m,1);
    A=zeros(m,m);
    b=zeros(m,1)
    for j=1:m  % row
        for i=1:m % col
            item_i=0;
            for k=1:n
               item_i = item_i+power(X(k),i-1) * power(X(k),j-1);
            end
            A(j,i) = item_i;
        end
        item_j = 0;
        for k=1:n
            item_j = item_j + Y(k)*power(X(k),j-1);
        end
        b(j) = item_j;
    end
    A
    b
    W = A\b;
end

  

