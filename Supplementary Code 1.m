clear all
 
format long
iptsetpref('ImshowBorder','tight');
 
File_Start_Number=700;
File_End_Number=999; 
Full_data=[];
MxP_data(1)=380;
MxP_index= 1;
ratio = 0.0303 ;
 
for idx=File_Start_Number:File_End_Number
    LI=xlsread("astm_TrueLI2.xlsx");
    
    %%% File Read
    
    File_Index_Number=int2str(idx);
    
    while length(File_Index_Number)<6
          File_Index_Number=['0' File_Index_Number];
    end
    
    File_Index_Number=['ff' File_Index_Number '.jpg'];
    img=imread('ff000068.jpg'); 
    img2=imread(File_Index_Number);
       
    %%% File Process
    
    img_double=im2double(img);
    b=img_double(1:392,44:435);
    img2_double=im2double(img2);
    c=img2_double(1:392,44:435);
    
    x=img2_double(1:392,44:435);
    img_diff=c-b;
    G = fspecial('gaussian',[30 30],30); 
    Filtering_Img=imfilter(img_diff,G,'same');
    
    
    %%% Get Maximum Point
   
    mask=Filtering_Img(166:182,44:380);
    MxP_value=max(mask(:));
    [MxP_y,MxP_x]=find(Filtering_Img==MxP_value);
    
    if MxP_x>=MxP_data(MxP_index)
        MxP_x=MxP_data(MxP_index);
        MxP_value=Filtering_Img(MxP_y,MxP_x);
    end
    
    MxP_data=[MxP_data; MxP_x];
    MxP_index=MxP_index+1;
     %%% Draw Contour
    gap1=round(0.2/ratio);
    gap2=round(0.4/ratio)-gap1;
    Switch=-1; 
    Contour_Gap=0; 
    Contour_Number=1;
    S=[];
    po=[];
    
    imshow(x) 
    hold on;
    
    for Draw_Contour=1:1:5  
           
        if Switch>0
               Contour_Gap=Contour_Gap+gap2;
               if MxP_x-Contour_Gap<=0 
                   Contour_Gap=1;
               end
               
            else 
               Contour_Gap=Contour_Gap+gap1;
               if MxP_x-Contour_Gap<=0 
                   Contour_Gap=1; 
               end
        end
        
        Contour_value=Filtering_Img(MxP_y,MxP_x-Contour_Gap);
        [C,h]=contour(Filtering_Img,[Contour_value,Contour_value]);        
        
        diff=50;
        CPx=find(C(1,:)<=(MxP_x-Contour_Gap+diff) & C(1,:)>=(MxP_x-Contour_Gap-diff));
        CPy=find(C(2,:)<=MxP_y+diff & C(2,:)>=MxP_y-diff);
        CPx=unique(CPx);
        CPy=unique(CPy);
        CP=intersect(CPx,CPy);
        angle=180*atan2(MxP_y-C(2,CP),MxP_x-C(1,CP))/pi;
 
          for tw=1:1:length(angle)
            if abs(angle(tw))<
                line=sqrt(((MxP_x-C(1,CP(tw)))^2+(MxP_y-C(2,CP(tw)))^2));
                S=[S; C(1,CP(tw))' C(2,CP(tw))' Contour_Number line angle(tw)];
            end
         end
         
        po=[po; Filtering_Img(MxP_y,MxP_x-Contour_Gap)];
        Contour_Number=Contour_Number+1;
        Switch=-Switch;
    end
    
    fig = gcf;
    scatter(MxP_x,MxP_y,20,'w','filled');
    saveas(fig,idx+".bmp"); 
    close;
    
    sa=[]; 
    for a=1:1:length(po)
        
        load_gap=0.001;
        load=find(LI(:,2)>=po(a)-load_gap & LI(:,2)<po(a)+load_gap);
        
        while isempty(load)
            load_gap=load_gap*2;
            load=find(LI(:,2)>=po(a)-load_gap & LI(:,2)<po(a)+load_gap);  
        end
        mx=length(load);
        for l=1:1:length(load)
            if abs(po(a)-LI(load(mx),2))>abs(po(a)-LI(load(l),2))
                mx=l;
            end
        end
        sa=[sa; a LI(load(mx),2) LI(load(mx),1)];
    end
  
    n=4; %% strain hardening component of the sample
    So=22.8; %% Yield stress
    v=0.33; %% Poisson ratio of the sample
    al=0.26; %%% Ramberg-Osgood constant
    In=3.75; %%% constant
    E=2.8*1000; %% Young’s Modulus the of sample
        n4_data=xlsread('Book4.xlsx','A2:E1017');
 
    Th1=n4_data(:,1);
    R1=n4_data(:,5);
    Thp1=0;
    
    for i=1:length(Th1)
        Thp1(i)=-Th1(i);
        Rp1(i)=R1(i);
    end
    
    Fth1=flip(Thp1);
    FRp1=flip(Rp1);
    Total_th1=[Th1;Fth1']; %%%5 theta in degree
    Deg_Rad1=deg2rad(Total_th1); %%% theta in radian
    Total_R1=[R1;FRp1']; %%% dimensionless effective stress
    RoDeg=round(Total_th1(1:end-1),1);
    RoR=(Total_R1(1:end-1));
    
    C_N = 4; %% contour that is considered for the calculation
    
    aa=find(S(:,3)==C_N);
    m=round(S(aa,5));
    
    for j=1:length(m)
    [R(j),C(j)]=(find(RoDeg==m(j)));
    EffS(j)=RoR(R(j));
    end
    
    EffS';
    Eff=[EffS'];
 
    bb=unique(aa);
    L=sa(C_N,3);
    C_inten=sa(C_N,2);
    nn=length(m);
    ww=ones(nn,1).*L;
    rr=S(bb,4);
    w=[ww];
    
    for s=1:50
        K1(1)=1;
        dK1(1)=0;
        K1(s+1)=K1(s)+dK1(s);
         for u=2:length(w)+1
            r(u-1)=rr(u-1).*ratio*10.^(-3);
            f(u-1)=So*K1(s+1)*1./(r(u-1).^(1./(n+1))).*Eff(u-1)-w(u-1);
            a2(u-1)=So*1./(r(u-1).^(1/(n+1))).*Eff(u-1);
        end
        b1=a2';
        b4=f';
        p5=-[b1];
        p6=inv(p5'*p5)*(p5'*b4);
        dK1(s+1)=p6(1);
    end
    
    p1=[K1' ]; % stress intensity factor
    p=[ dK1' ]; % change in stress intensity factor
     K=p1(length(p1));
    J=((al*So.^2*In*10.^6)*K.^(n+1))./E;
    cracklength = 15+((379-(MxP_data(MxP_index)+43))*ratio) ; %% determination of crack length
    
    Full_data=[Full_data; idx K J cracklength MxP_value C_inten];
    clearvars -except Full_data index MxP_data MxP_index ratio C_inten; 
   end
 
csvwrite("final data.csv",Full_data)
