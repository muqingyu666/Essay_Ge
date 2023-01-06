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
	dim		= dimsizes(eof)
; ===========================================================================
	faero	= addfile("/mnt/e/fig/AOD_bilinear_2010-2020.nc","r")
	AOD		= faero->AOD_bilinear(:,{latS:latN+0.5},{lonL:lonR+0.5})
; ===========================================================================   
	fCF		= addfile("/mnt/e/fig/第五版/fig.5/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20100101-20210131.nc","r")
	Var1	= fCF->toa_net_clr_daily(0:4017,{latS:latN+0.5},{lonL:lonR+0.5})
	Var2	= fCF->toa_net_all_daily(0:4017,{latS:latN+0.5},{lonL:lonR+0.5})
	Var		= Var1-Var2
	copy_VarCoords(Var1, Var)
; ===========================================================================
	AOD	= where(AOD.lt.0.06,AOD@_FillValue,AOD)
	x	= AOD
	y	= Var
; ===========================================================================
	n_line	= 6
	box_line	= new( (/ n_line+1 /), float )
	
    box_line(0)			= min(ndtooned(eof))
	do g = 1,n_line-1
		box_line(g)		= percent_to_value( eof, 100./n_line*g )
	end do
    box_line(n_line)	= max(ndtooned(eof))
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
				if(number.ge.10)then
					flag 				:= regline(x(:,m,n),y(:,m,n))
					rc(0,m,n)		 = flag
  				tval := flag@tval   ;t-statistic of rc
  				df   := flag@nptxy  ;自由度
  				b		 := tval    ; b must be same size as tval (and df)
  				b     = 0.5
  				prob := betainc(df/(df+tval^2),df/2.0,b)       ; prob(nlat,nlon)
					rc_num(0,m,n)	= prob
				else
					rc(0,m,n)		= y@_FillValue
					rc_num(0,m,n)	= rc_num@_FillValue
				end if
			end if
			
			index	:= ind(eof(:,m,n).gt.0.76.and..not.ismissing(x(:,m,n)))
			numer	:= num(ind(eof(:,m,n).gt.0.76.and..not.ismissing(x(:,m,n))))
			if(numer.ge.10)then
				flag 				:= regline(x(index,m,n),y(index,m,n))
				rc(1,m,n)		= flag
  			tval := flag@tval   ;t-statistic of rc
  			df   := flag@nptxy  ;自由度
  			b		 := tval    ; b must be same size as tval (and df)
  			b     = 0.5
  			prob := betainc(df/(df+tval^2),df/2.0,b)       ; prob(nlat,nlon)
				rc_num(1,m,n)	= prob
			else
				rc(1,m,n)		= y@_FillValue
				rc_num(1,m,n)	= rc_num@_FillValue
			end if

			index1	:= ind(eof(:,m,n).gt.-0.03.and.eof(:,m,n).lt.0.5.and..not.ismissing(x(:,m,n)))
			numer1	:= num(ind(eof(:,m,n).gt.-0.03.and.eof(:,m,n).lt.0.5.and..not.ismissing(x(:,m,n))))
			if(numer1.ge.10)then
				flag 				:= regline(x(index1,m,n),y(index1,m,n))
				rc(2,m,n)		= flag
  			tval := flag@tval   ;t-statistic of rc
  			df   := flag@nptxy  ;自由度
  			b		 := tval    ; b must be same size as tval (and df)
  			b     = 0.5
  			prob := betainc(df/(df+tval^2),df/2.0,b)       ; prob(nlat,nlon)
				rc_num(2,m,n)	= prob
			else
				rc(2,m,n)		= y@_FillValue
				rc_num(2,m,n)	= rc_num@_FillValue
			end if
			
			index2	:= ind(eof(:,m,n).lt.-0.41.and..not.ismissing(x(:,m,n)))
			numer2	:= num(ind(eof(:,m,n).lt.-0.41.and..not.ismissing(x(:,m,n))))			;.and..not.ismissing(y(:,m,n))
			if(numer2.ge.10)then
				flag 				:= regline(x(index2,m,n),y(index2,m,n))
				rc(3,m,n)		= flag
  			tval := flag@tval   ;t-statistic of rc
  			df   := flag@nptxy  ;自由度
  			b		 := tval    ; b must be same size as tval (and df)
  			b     = 0.5
  			prob := betainc(df/(df+tval^2),df/2.0,b)       ; prob(nlat,nlon)
				rc_num(3,m,n)	= prob
			else
				rc(3,m,n)		= y@_FillValue
				rc_num(3,m,n)	= rc_num@_FillValue
			end if
		end do
	end do  
	copy_VarCoords(Var(0:3,:,:), rc)
	copy_VarCoords(Var(0:3,:,:), rc_num)
;************************************************
; plotting parameters 
;************************************************
	wks		= gsn_open_wks("x11", "CRF_AOD_PC1_space_regress_4plot")        ; send graphics to PNG file
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
	;res@gsnSpreadColorStart		= 30
	;res@gsnSpreadColorEnd			= 100
	res@cnLevelSelectionMode  = "ManualLevels"   ; set manual contour levels
	res@cnMinLevelValF        = -400;-100           ; set min contour level
	res@cnMaxLevelValF        = 400;250           ; set max contour level
	res@cnLevelSpacingF       = 20;10           ; set contour interval 
	res@gsnLeftString		  = "~F34~6~N~"+"~F25~CRF~N~"+"/"+"~F34~6~N~"+"~F25~AOD~N~"
    res@gsnLeftStringFontHeightF    = 0.03
    res@gsnRightStringFontHeightF   = 0.03
	RightString				  = (/" ","PC1>0.76","-0.03<PC1<0.5","PC1<-0.41"/)
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
	
;显著性打点
  res2 										= True
  res2@gsnDraw 						= False
  res2@gsnFrame 					= False
  res2@cnFillOn 					= True
  res2@cnLinesOn 					= False
  res2@cnLineLabelsOn 		= False
  res2@cnInfoLabelOn 			= False
  res2@lbLabelBarOn 			= False
  res2@cnMonoFillPattern 	= False
  res2@cnLevelSelectionMode = "ExplicitLevels"
  res2@cnLevels 					= (/0.01/)                       ;; set to significance level
  res2@cnFillPatterns 		= (/17,-1/)
  res2@cnFillColors 			= (/1,0/)
  res2@cnFillScaleF     	= 1.2
  res2@cnFillDotSizeF			= 0.003
  res2@gsnLeftString 			= ""
	
	do i = 0,3
		res@gsnRightString	= RightString(i)
		plot(i) 	= gsn_csm_contour_map_ce(wks,rc(i,:,:),res)
		poly(i)		= gsn_add_polyline(wks,plot(i),mrb_lon,mrb_lat,tpres)
  	porb(i) 	= gsn_csm_contour(wks,rc_num(i,:,:),res2)
		overlay(plot(i),porb(i))
	end do
	
	; lnres					= True
	; lnres@gsLineThicknessF	= 5.0
	; lnres@gsLineColor		= "red"
	; txres					= True
	; txres@txFontHeightF		= 0.02
	; txres@txJust			= "CenterCenter"
	; text					= (/"C","B","A"/)
	; xpos	= (/(/94,84,84,94,94/),(/92,87,87,92,92/),(/96,86,86,96,96/)/)
	; ypos	= (/(/29,29,30,30,29/),(/32,32,34,34,32/),(/37,37,38,38,37/)/)
	; xsit	= (/89,89.5,91/)
	; ysit	= (/29.5,33,37.5/)
	; n = 0
	; do oo = 0,3
		; do pp = 0,2
			; poy(n)		= gsn_add_polyline(wks,plot(oo),xpos(pp,:),ypos(pp,:),lnres)
			; dum(n)		= gsn_add_text(wks,plot(oo),text(pp),xsit(pp),ysit(pp),txres)
			; n = n + 1
		; end do
	; end do
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
	resP@gsnPanelXWhiteSpacePercent	= 3.			;相邻子图X轴方向大小
	resP@gsnPanelYWhiteSpacePercent	= 3.			;相邻子图Y轴方向大小
	resP@gsnPanelMainFontHeightF	= .018			;主标题字体大小
	;resP@gsnPanelScalePlotIndex		= 2
	resP@gsnPanelMainString			= ""		;主标题
	gsn_panel(wks,(/plot/),(/2,2/),resP)     ; now draw as one plot
			
  end
