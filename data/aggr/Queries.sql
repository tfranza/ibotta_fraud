-- Multiple Tables Join 
select *
from card cc
join disp dis on dis.disp_id = cc.disp_id 
join account ac on ac.account_id = dis.account_id 
join loan l on l.account_id = ac.account_id 
join trans tr on tr.account_id = l.account_id 

-- Trans Table
select *
from trans t2 