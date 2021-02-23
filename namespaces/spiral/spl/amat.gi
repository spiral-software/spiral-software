
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F SPLAMat( <amat> )
#F   returns an spl corresponding to <amat>.
#F

SPLAMat := function ( A )
  if not IsAMat(A) then
    Error("<A> must be an amat");
  elif A.char <> 0 then
    Error("<A> has char <> 0");
  fi;

  # catch identity case
  if IsIdentityMat(A) then
    return Checked(A.dimensions[1]=A.dimensions[2], I(A.dimensions[1]));
  fi;

  # dispatch on types
  if A.type = "perm" then

    return Perm(A.element, A.dimensions[1]);

  elif A.type = "mon" then
    
    # catch diagonal case
    if A.element.perm = () then
      return Diag(A.element.diag);
    fi;
    return Perm(A.element.perm, A.dimensions[1]) * Diag(A.element.diag);

  elif A.type = "mat" then

    # catch DFT case
    if IsBound(A.isDFT) and A.isDFT = true then
      return spiral.transforms.DFT(A.dimensions[1]);
    fi;

    # catch RDFT case
    if  IsBound(A.isRDFT) and A.isRDFT = true then
      if A.RDFTinfo[1] = 1 then
        if A.RDFTinfo[2] = "not transposed" then
          return spiral.transforms.SRDFT(A.dimensions[1]);
        else # transposed
          return spiral.transforms.SRDFT(A.dimensions[1]).transpose();
        fi;
      else # A.RDFTinfo[1] = 3
        if A.RDFTinfo[2] = "not transposed" then
          return spiral.transforms.SRDFT3(A.dimensions[1]);
        else # transposed
          return spiral.transforms.SRDFT3(A.dimensions[1]).transpose();
        fi;
      fi;
    fi;

    # catch rotation case
    if IsBound(A.isRotation) and A.isRotation = true then
      return Rot(A.angle);
    fi;

    return Mat(A.element);

  elif A.type = "scalarMultiple" then

    if IsIdentityMat(A.element) then
      return 
        Diag(
          List([1..A.dimensions[1]], i -> A.scalar)
        );
    else
      return 
        Compose(
          Diag(
            List([1..A.dimensions[2]], i -> A.scalar)
          ),
          SPLAMat(A.element)
        );
    fi;

  elif A.type = "product" then

    return Compose(List(A.factors, SPLAMat));

  elif A.type = "power" then
    
    # no negative exponents
    if A.exponent < 0 then
      Error("<A>.exponent must not be negative");
    fi;
    return Compose(Replicate(A.exponent, SPLAMat(A.element)));

  elif A.type = "conjugate" then

    return Conjugate(SPLAMat(A.element), SPLAMat(A.conjugation));

  elif A.type = "directSum" then

    return DirectSum(List(A.summands, SPLAMat));

  elif A.type = "tensorProduct" then   

    return Tensor(List(A.factors, SPLAMat));

  else

    Error("unrecognized type of amat <A>");

  fi;
end;


