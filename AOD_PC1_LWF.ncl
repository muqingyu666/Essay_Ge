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
;=====================================================================================
begin	
	latS   =  25.
	latN   =  40. 
	lonL   =  70. 
	lonR   =  105.
;=============================气溶胶部分==============================================	
	faero	= addfile("/mnt/e/fig/AOD_bilinear_2010-2020.nc","r")
	AOD		= faero->AOD_bilinear(:,{latS:latN+0.5},{lonL:lonR+0.5})	
;=============================辐射部分==================================================	
	fCF		= addfile("/mnt/e/fig/第五版/fig.5/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20100101-20210131.nc","r")
	Var1	= fCF->toa_lw_clr_daily(0:4017,{latS:latN+0.5},{lonL:lonR+0.5})
	Var2	= fCF->toa_lw_all_daily(0:4017,{latS:latN+0.5},{lonL:lonR+0.5})
	Var		= Var1-Var2
;=============================气象场部分==============================================
	fmete	= addfile("/mnt/e/fig/EOF_T300_RH300_U300_w300_se300_lunkuonei.nc","r")
	eof		= fmete->EOF(0,:,{latS:latN+0.5},{lonL:lonR+0.5})
;=====================================================================================
	tAOD		= ndtooned(AOD)
	tVar		= ndtooned(Var)
	teof	= ndtooned(eof)
	index	= ind(.not.ismissing(tAOD).and..not.ismissing(tVar).and..not.ismissing(teof).and.tVar.ne.0.and.tAOD.ge.0.06)
	ttAOD	= tAOD(index)
	ttVar	= tVar(index)
	tteof	= teof(index)

	 printMinMax(ttVar,True)
	; printMinMax(tteof,True)
	; printMinMax(ttAOD,True)
	; printVarSummary(tteof)
	;=============================box==============================================
	n_box	= 6
	n_line	= 6
	data		= new((/ n_box,n_line /),float)
	XB			= new((/ n_box,n_line /),float)
	box_line	= new( (/ n_line+1 /), float )
	x_box		= new( (/ n_box+1 /), float )
	
    box_line(0)			= min(tteof)
	do g = 1,n_line-1
		box_line(g)		= percent_to_value( tteof, 100./n_line*g )
	end do
    box_line(n_line)	= max(tteof)
	decimal_line		= decimalPlaces(box_line,2,True)
	; print(box_line)
	
    x_box(0) 		= min(ttAOD)
	; x_box(1)		= percent_to_value( ttAOD, 100./50 )
	do h = 1,n_box-1
		x_box(h)	= percent_to_value( ttAOD, 100./n_box*h )
	end do
	; x_box(n_box-1)		= percent_to_value( ttAOD, 100./50*49 )
    x_box(n_box)	= max(ttAOD)
	decimal_box		= decimalPlaces(x_box,2,True)
	do m = 0, n_box-1
		do n = 0, n_line-1
			if(m.eq.n_box-1)then
				index	:= ind(ttAOD.ge.x_box(m).and.ttAOD.le.x_box(m+1).and.tteof.ge.box_line(n).and.tteof.lt.box_line(n+1))
			else if(n.eq.n_line-1)then
				index	:= ind(ttAOD.ge.x_box(m).and.ttAOD.lt.x_box(m+1).and.tteof.ge.box_line(n).and.tteof.le.box_line(n+1))
			else if(m.eq.n_box-1.and.n.eq.n_line-1)then
				index	:= ind(ttAOD.ge.x_box(m).and.ttAOD.le.x_box(m+1).and.tteof.ge.box_line(n).and.tteof.le.box_line(n+1))
			else
				index	:= ind(ttAOD.ge.x_box(m).and.ttAOD.lt.x_box(m+1).and.tteof.ge.box_line(n).and.tteof.lt.box_line(n+1))
			end if
			end if
			end if
			data(m,n)	= avg(ttVar(index))
			XB(m,n)		= avg(ttAOD(index))
		end do
	end do
	; print(XB)
	rc			= new((/ n_line, 3 /),float)	
	do l = 0, n_line-1
		indices	:= ind(tteof.ge.box_line(l).and.tteof.lt.box_line(l+1))
		reg		:= regline(ttAOD(indices),ttVar(indices))
		rc(l,0)	 = reg
		rc(l,1)	 = reg@yintercept		
		if(abs(reg@tval).lt.0.6745)
			rc(l,2)	 = 0
		else if(abs(reg@tval).ge.0.6745.and.abs(reg@tval).lt.1.6449)
			rc(l,2)	 = 1
		else
			rc(l,2)	 = 2
		end if
		end if
	end do
	; print(rc)
;=============================绘图部分==============================================
	wks		= gsn_open_wks ("x11","AOD_PC1_LWF")  
	plot	= new(n_line,"graphic")	; send graphics to PNG file
	res						= True                     ; plot mods desired
    res@gsnDraw				= False       ; don't draw yet
    res@gsnFrame			= False       ; don't advance frame yet
	;res@tiMainString			= "Scatter Plot"
	res@tiXAxisOn				= True
	res@tiYAxisOn				= True
	res@tiXAxisString			= "AOD"
	res@tiYAxisString			= "LWCRF(W m~S~-2~N~)"
	res@tiXAxisFontHeightF		= 0.025
	res@tiXAxisFontThicknessF	= 2.
	res@tiYAxisFontHeightF		= 0.025
	res@tiYAxisFontThicknessF	= 2.
    res@trXMinF					= 0;-3. 
    res@trXMaxF					= 0.35;-0.9
    res@trYMinF					= -20;2    
    res@trYMaxF					= 70;4.1
	res@tmXBMode						= "Manual"
	res@tmXBTickSpacingF		= 0.1
	;res@tmXBPrecision				= 1
	res@tmXBMinorPerMajor		= 4
	res@tmXBLabelDeltaF		  	= -0.3
	res@tmXBLabelFontHeightF	= 0.021
	res@tmXBLabelFontThicknessF	= 0.05
	res@tmYLMode				= "Manual"
	res@tmYLTickSpacingF		= 20
	res@tmYLMinorPerMajor		= 3
	res@tmYLLabelFontHeightF	= 0.021
	res@tmYLLabelFontThicknessF	= 0.05
	res@tmYLLabelDeltaF		 	= -0.3
	
  res@tmXMajorGrid					= True
  res@tmXMajorGridLineDashPattern	= 3
  res@tmXMajorGridLineColor			= "gray"
  res@tmXMajorGridThicknessF		= 2
  res@tmYMajorGrid					= True
  res@tmYMajorGridLineDashPattern	= 3
  res@tmYMajorGridLineColor			= "gray"
  res@tmYMajorGridThicknessF		= 2
  res@tmXUseBottom				= False      ; Keep top axis independent of bottom.
  res@tmYUseLeft				= False
  
  res@tmXTOn             	   = False       ; Don't draw labels for
  res@tmXTLabelsOn      	   = False
  
  res@tmYROn             	   = False       ; Don't draw labels for
  res@tmYRLabelsOn      	   = False

	res@xyMarkLineMode			= "MarkLines"
	xy_marker					= (/11,0,6,9,13,16/)               ; 3 different markers
	xy_marker_col				= (/"blue","NavyBlue","Orange","ForestGreen","Black","red"/) ; 3 different colors
	xy_line_thick				= (/5,5,5,5,5,5/)
	xy_line_col					= (/"blue","NavyBlue","Orange","ForestGreen","Black","red"/)
	xy_dash						= (/0,1,2,3,4,5/)
	xy_legend_sign				= (/"<PC1<"/)
	xy_legend_space_L			= (/"","",""," "," "," "/)
	xy_legend_space_R			= (/"     ","     ","      ","      ","      ","      "/)
	xy_legend_value				= decimal_line                    ; nice tick mark labels
	prob						= (/"**","**","**","**","**","*"/)
	do	i = 0,n_line-1
		res@xyMarkers				= xy_marker(i)
		res@xyMarkerColors			= xy_marker_col(i)
		res@xyLineThicknesses		= xy_line_thick(i)
		res@xyLineColors			= xy_line_col(i)
		res@xyDashPatterns			= xy_dash(i)
		; if(i.eq.0.or.i.eq.1)then
			; plot(i)  = gsn_csm_xy(wks,XB(3:n_box-1,i),data(3:n_box-1,i),res)
		; else if(i.eq.2.or.i.eq.4)
			; plot(i)  = gsn_csm_xy(wks,XB(1:n_box-1,i),data(1:n_box-1,i),res)
		; else
			plot(i)  = gsn_csm_xy(wks,XB(:,i),data(:,i),res) ; create plot
		; end if
		; end if
		res@tiXAxisOn				= False
		res@tiYAxisOn				= False
		res@tmXBOn             		= False
		res@tmXBLabelsOn      		= False
		res@tmXTOn             		= False
		res@tmXTLabelsOn      		= False
		res@tmXMajorGrid			= False
		res@tmYLOn             		= False
		res@tmYLLabelsOn      		= False
		res@tmYROn             		= False
		res@tmYRLabelsOn      		= False
		res@tmYMajorGrid			= False
	end do
	
	genres                     = True
	textres                    = True
	lineres                    = True
	genres@XPosPercent         = 4;72					; change legend X-axis position
	genres@YPosPercent         = 23
	genres@ItemSpacePercent	   = 4;2.52          			; expressed as %, 0->100, space between legend items (Default = 5.)
	genres@LineLabelWhiteSpacePercent = 2 			; expressed as %, 0->100, space between line and label (Default = 3.
	textres@lgItemCount        = 15
	textres@lgLabelFontHeights = 0.015                           ; label font heights
	textres@lgLabelOpacity	   = 0.99
	textres@lgLabels           = xy_legend_value(0:5)+xy_legend_space_L+xy_legend_sign +xy_legend_value(1:6) \; reverse label order for legend
									+ xy_legend_space_R + "LWCRF = " + sprintf("%4.1f",rc(:,0)) + " * " + "AOD + "+ sprintf("%4.1f",rc(:,1))+prob
	lineres@lgLineThicknesses  = 5;4.5
	lineres@lgLineColors       = xy_line_col     	; reverse color order for legend
	lineres@lgDashIndexes      = xy_dash   			; reverse pattern order for legend
	lineres@LineLengthPercent  = 10
	dum = simple_legend(wks, plot(n_line-1), genres, lineres, textres)
	draw(plot)
	frame(wks)
end