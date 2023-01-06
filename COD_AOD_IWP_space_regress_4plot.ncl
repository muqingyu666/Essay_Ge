undef("percent_to_value")
function percent_to_value( \
  i_data           : numeric, \
  i_percentiles[*] : numeric  \
)
local None
begin

  retVal = new( dimsizes(i_percentiles), float )

  data1d = ndtooned( i_data )
  notMissing = data1d( ind(.not.ismissing(data1d) ) )

  qsort(notMissing)

  do p = 0, dimsizes(i_percentiles)-1
;   pInd = round( i_percentiles(p) * .01 * dimsizes(notMissing) + 0.5, 3 ) -1
;   pInd = where( pInd.ge.dimsizes(notMissing), dimsizes(notMissing)-1, pInd )
    floatInd = i_percentiles(p) * .01 * dimsizes(notMissing) - 0.5
    floorInd = toint( floor(floatInd) )
    floorInd = where( floorInd.lt.0, 0, floorInd )
    ceilInd = toint( ceil(floatInd) )
    ceilInd = where( ceilInd.ge.dimsizes(notMissing), \
        dimsizes(notMissing)-1, ceilInd )
;   print(pInd + " " + dimsizes(notMissing))
    if( ceilInd.eq.floorInd ) then
      retVal(p) = notMissing(floorInd)
    else
      retVal(p) = notMissing(floorInd) * ( ceilInd - floatInd ) \
          + notMissing(ceilInd) * ( floatInd - floorInd )
    end if
  end do

  return(retVal)
end ; percent_to_value
begin
;************************************************
; create pointer to file and read in temperature
;************************************************
	latS   =  25.
	latN   =  40. 
	lonL   =  70. 
	lonR   =  105.
;===========================================================================
	fmete	= addfile("/mnt/e/fig/EOF_T300_RH300_U300_w300_se300_lunkuonei.nc","r")
	eof		= fmete->EOF(0,:,{latS:latN+0.5},{lonL:lonR+0.5})
; ===========================================================================   
	ff		= addfile("/mnt/e/fig/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20100101-20201231.nc","r")
	IWP		= ff->iwp_high_daily(:,{latS:latN+0.5},{lonL:lonR+0.5})
	Var		= ff->cldtau_high_daily(:,{latS:latN+0.5},{lonL:lonR+0.5})
	dim		= dimsizes(IWP)	
; ===========================================================================
	faero	= addfile("/mnt/e/fig/AOD_bilinear_2010-2020.nc","r")
	AOD		= faero->AOD_bilinear(:,{latS:latN+0.5},{lonL:lonR+0.5})
; ===========================================================================
	Var	= where(Var.eq.0,Var@_FillValue,Var)
	AOD	= where(AOD.lt.0.06,AOD@_FillValue,AOD)
	x	= log(AOD)
	y	= log(Var)
; ===========================================================================
	n_line	= 6
	box_line	= new( (/ n_line+1 /), float )
	
    box_line(0)			= min(ndtooned(IWP))
	do g = 1,n_line-1
		box_line(g)		= percent_to_value( IWP, 100./n_line*g )
	end do
    box_line(n_line)	= max(ndtooned(IWP))
	decimal_line		= decimalPlaces(box_line,2,True)
;************************************************                                ; no longer needed
; create x and calculate the regression coefficients (slopes, trends)    
;************************************************ 
	rc			= new((/ 4, dim(1), dim(2) /),float,-999)
	rc_num		= new((/ 4, dim(1), dim(2) /),float,-999)
	do m = 0, dim(1)-1
		do	n = 0, dim(2)-1
			if(.not.all(ismissing(eof(:,m,n))))
				number	:= num(ind(.not.ismissing(x(:,m,n))))
				if(number.ge.3)then
					rc(0,m,n)		= regline(x(:,m,n),y(:,m,n))
					rc_num(0,m,n)	= number
				else
					rc(0,m,n)		= y@_FillValue
					rc_num(0,m,n)	= rc_num@_FillValue
				end if

				index	:= ind(IWP(:,m,n).gt.244.9.and..not.ismissing(x(:,m,n)))
				numer	:= num(ind(IWP(:,m,n).gt.244.9.and..not.ismissing(x(:,m,n))))
				if(numer.ge.3)then
					rc(1,m,n)		= regline(x(index,m,n),y(index,m,n))
					rc_num(1,m,n)	= numer
				else
					rc(1,m,n)		= y@_FillValue
					rc_num(1,m,n)	= rc_num@_FillValue
				end if

				index1	:= ind(IWP(:,m,n).gt.31.2.and.IWP(:,m,n).lt.120.2.and..not.ismissing(x(:,m,n)))
				numer1	:= num(ind(IWP(:,m,n).gt.31.2.and.IWP(:,m,n).lt.120.2.and..not.ismissing(x(:,m,n))))
				if(numer1.ge.3)then
					rc(2,m,n)		= regline(x(index1,m,n),y(index1,m,n))
					rc_num(2,m,n)	= numer1
				else
					rc(2,m,n)		= y@_FillValue
					rc_num(2,m,n)	= rc_num@_FillValue
				end if
				
				index2	:= ind(IWP(:,m,n).lt.12.6.and..not.ismissing(x(:,m,n)))
				numer2	:= num(ind(IWP(:,m,n).lt.12.6.and..not.ismissing(x(:,m,n))))			;.and..not.ismissing(y(:,m,n))
				if(numer2.ge.3)then
					rc(3,m,n)		= regline(x(index2,m,n),y(index2,m,n))
					rc_num(3,m,n)	= numer2
				else
					rc(3,m,n)		= y@_FillValue
					rc_num(3,m,n)	= rc_num@_FillValue
				end if
			end if
		end do
	end do
	rc@long_name = "~F34~6~N~"+"~F25~lnCOD~N~"+"/"+"~F34~6~N~"+"~F25~lnAOD~N~"
	rc@units     = " "   
	copy_VarCoords(Var(0:3,:,:), rc)
	copy_VarCoords(Var(0:3,:,:), rc_num)
;==============================空间分布显著性检验================================================
	t_rc		= new((/ 4, dim(1), dim(2) /),float,-999)
	p_rc		= new((/ 4, dim(1), dim(2) /),float,-999)
	Rxt_rc		= new((/ 4, dim(1), dim(2) /),float,-999)
	Rxt_rc_max	= new((/ 4 /),float,-999)
	Rxt_rc_min	= new((/ 4 /),float,-999)
	do s = 0,3
		t_rc(s,:,:)		= rc(s,:,:)*sqrt(rc_num(s,:,:)/(1.-rc(s,:,:)^2)) ;t检验
		p_rc(s,:,:)		= student_t(t_rc(s,:,:), rc_num(s,:,:)) ;两侧student-t分布概率值		
		
		Rxt_rc(s,:,:)	= where(p_rc(s,:,:).le.0.05, rc(s,:,:), rc@_FillValue)
		Rxt_rc_max(s)	= max(Rxt_rc(s,:,:))
		Rxt_rc_min(s)	= min(Rxt_rc(s,:,:))
	end do
	copy_VarCoords(Var(0:3,:,:), Rxt_rc)
;************************************************
; plotting parameters 
;************************************************
	wks		= gsn_open_wks("x11", "HOD_AOD_space_regress_4plot")        ; send graphics to PNG file
	plot	= new(4,"graphic")
	poly	= new(4,"graphic")
	porb	= new(4,"graphic")
	poy		= new(12,"graphic")
	dum		= new(12,"graphic")

	res                             = True
	res@gsnDraw                     = False
	res@gsnFrame                    = False
	res@gsnAddCyclic                = False      ;data is not cyclic
	res@lbLabelBarOn	      		= False
	res@cnFillOn                    = True         ; turn on color fill
	res@cnFillMode            		= "RasterFill"
	res@cnLinesOn                   = False
	res@cnLineLabelsOn              = False      ;turn off cn line labels
	res@cnInfoLabelOn               = False      ;turn off contour information label
	res@mpFillOn                    = False        ; turn off map fill
	res@mpMinLatF                   = latS
	res@mpMaxLatF                   = latN
	res@mpMinLonF                   = lonL
	res@mpMaxLonF                   = lonR
	res@mpCenterLonF                = (lonL+lonR)/2
	res@mpGridAndLimbOn             = True
	res@mpGridLineThicknessF        = 0.5
	res@mpGridLineDashPattern       = 2
	res@mpGridSpacingF              = 5.
	res@mpOutlineOn                 = True      
	res@gsnMaximize           = True             ; make large
	res@cnFillPalette         = "ViBlGrWhYeOrRe" ; set color map

	res@cnLevelSelectionMode  = "ManualLevels"   ; set manual contour levels
	res@cnMinLevelValF        =  -1.5           ; set min contour level
	res@cnMaxLevelValF        =   1.5           ; set max contour level
	res@cnLevelSpacingF       =   0.1           ; set contour interval 
	RightString				  = (/" ","IWP>244.9","31.2<IWP<120.2","IWP<12.6"/)
    res@gsnLeftStringFontHeightF    = 0.03
    res@gsnRightStringFontHeightF   = 0.03
	res@tiMainString          = ""
    res@tmXBLabelFontHeightF        = 0.03    ;font height of tick labels
    res@tmYLLabelFontHeightF        = 0.03
;添加青藏高原轮廓
	f            = addfile("/mnt/d/DBATP/DBATP_Polygon.shp","r")
	mrb_lon      = f->x
	mrb_lat      = f->y	 
	tpres                    = True
	tpres@gsLineColor        = "Black"     ;设置廓线颜色
	tpres@gsLineThicknessF   = 3           ;设置廓线宽度
	
	do i = 0,3
		res@gsnRightString	= RightString(i)
		plot(i) = gsn_csm_contour_map_ce(wks,rc(i,:,:),res)
		poly(i)	= gsn_add_polyline(wks,plot(i),mrb_lon,mrb_lat,tpres)
	end do
;显著性打点绘图
	sres                 = True                                                                ; set up a second resource list           
	sres@gsnDraw         = False                                                        ;         do not draw the plot            
	sres@gsnFrame        = False                                                        ; do not advance the frame               
	sres@cnLineLabelsOn  = False                                                      ; do not use line labels                  
	sres@cnFillOn        = False                                                           ; color fill                              
	sres@cnLinesOn       = False                                                         ; do not draw contour lines               
	sres@cnInfoLabelOn   = False
	sres@cnLevelSelectionMode = "ExplicitLevels"           ; do not draw contour lines
	sres@cnLevels             = fspan(0,100,101)*0.01                     ; set the contour levels                                                                                                   
																																	
	opt1=True                                                                                                                       
	opt1@gsnShadeFillType = "pattern"                                                                                               
	; opt1@gsnShadeHigh     = 17                                                                                                      
	; opt1@gsnShadeLow      = 17
	opt1@gsnShadeMid     = 17 
	opt1@gsnShadeFillDotSizeF	= 0.003			;填充符号大小
	opt1@gsnShadeFillScaleF		= 1.2 		;填充符号密度                                                                           
	
	do q = 0,3
		porb(q)             = gsn_csm_contour(wks,Rxt_rc(q,:,:),sres)                                                                    
		porb(q)             = gsn_contour_shade(porb(q),Rxt_rc_min(q),Rxt_rc_max(q),opt1)    ;α=0.05                                              
		overlay(plot(q),porb(q))
	end do
;组图
	resP                     		= True         ; modify the panel plot
	resP@gsnMaximize         		= True         ; large format
	resP@gsnPanelLabelBar   		= True         ; add common colorbar
	resP@lbOrientation				= "horizontal"	;色标方向(vertical/horizontal)
	resP@lbLabelFontHeightF			= 0.02			;色标标签字体大小
	resP@lbLabelFontColor				= "black"		;色标标签字体颜色
	resP@lbLabelStride				= 5
	resP@pmLabelBarWidthF				= 0.8			;色标宽度
	resP@pmLabelBarHeightF			= 0.08			;色标高度
	resP@pmLabelBarParallelPosF 		= 0.			;水平移动色标
	resP@pmLabelBarOrthogonalPosF 	= 0.01			;垂直移动色标
	; resP@cnMinLevelValF 				= 0.			;色标最小值
	; resP@cnMaxLevelValF				= 40			;色标最大值
	; resP@cnLevelSpacingF				= 0.5			;色标间隔
	resP@gsnPanelBottom				= 0.0			;最下边图形绘制结束位置
	resP@gsnPanelLeft				= 0.0			;最左边图形绘制结束位置
	resP@gsnPanelRight				= 1.			;最右边图形绘制结束位置
	resP@gsnPanelTop				= 1.0			;最上边图形绘制结束位置
	resP@gsnPanelFigureStrings		= (/"a","b","c","d"/)
	resP@gsnPanelFigureStringsFontHeightF	= 0.015
	resP@amJust						= "topLeft"
	resP@gsnPanelXWhiteSpacePercent	= 3.			;相邻子图X轴方向大小
	resP@gsnPanelYWhiteSpacePercent	= 3.			;相邻子图Y轴方向大小
	resP@gsnPanelMainFontHeightF	= .018			;主标题字体大小
	;resP@gsnPanelScalePlotIndex		= 2
	resP@gsnPanelMainString			= ""		;主标题
	gsn_panel(wks,(/plot/),(/2,2/),resP)     ; now draw as one plot
			
  end
