function myBinaryPlot2D(example,class)
%Plot 2D points in the row of example with two different symbol for the
%class 1 and -1
hold on
for i=1:size(example,1)
    if(class(i)==1)
        plot(example(i,1),example(i,2),'b+')
    else
        plot(example(i,1),example(i,2),'ro')
    end
end
end

